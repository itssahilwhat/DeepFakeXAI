import os
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torchvision import transforms
from config import Config


class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, subset, transform=None, use_temporal=False, train_size=None, val_size=None, test_size=None):
        self.root_dir = root_dir
        self.subset = subset
        self.transform = transform
        self.use_temporal = use_temporal
        self.data = []

        # Set size limit based on subset
        self.size_limit = {
            'train': train_size,
            'valid': val_size,
            'test': test_size
        }.get(subset, None)

        if not os.path.isdir(root_dir):
            raise RuntimeError(f"❌ Dataset root not found: {root_dir}")

        cfg = Config.DATASET_CONFIGS.get(os.path.basename(root_dir), {})
        has_masks = cfg.get("has_masks", True)

        real_dir = os.path.join(root_dir, "real", subset)
        if os.path.isdir(real_dir):
            real_images = [f for f in os.listdir(real_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            for fname in real_images:
                img_path = os.path.join(real_dir, fname)
                self.data.append((img_path, None, 0))

        fake_root = os.path.join(root_dir, "fake")
        if os.path.isdir(fake_root):
            fake_images = []
            for method in os.listdir(fake_root):
                method_dir = os.path.join(fake_root, method)
                image_dir_1 = os.path.join(method_dir, "images", subset)
                mask_dir_1 = os.path.join(method_dir, "masks", subset)
                image_dir_2 = os.path.join(method_dir, subset)
                mask_dir_2 = os.path.join(method_dir.replace("images", "masks"), subset)

                final_img_dir = None
                final_mask_dir = None

                if os.path.isdir(image_dir_1):
                    final_img_dir = image_dir_1
                    final_mask_dir = mask_dir_1 if os.path.isdir(mask_dir_1) else None
                elif os.path.isdir(image_dir_2):
                    final_img_dir = image_dir_2
                    final_mask_dir = mask_dir_2 if os.path.isdir(mask_dir_2) else None

                if final_img_dir:
                    method_images = [f for f in os.listdir(final_img_dir) if
                                     f.lower().endswith((".jpg", ".jpeg", ".png"))]
                    # Sample every 3rd frame for temporal data
                    if self.use_temporal:
                        method_images = method_images[::3]
                    fake_images.extend([(method, f) for f in method_images])

            for method, fname in fake_images:
                img_path = os.path.join(fake_root, method, "images", subset, fname) if os.path.isdir(
                    os.path.join(fake_root, method, "images")) else os.path.join(fake_root, method, subset, fname)
                mask_path = os.path.join(fake_root, method, "masks", subset, fname) if os.path.isdir(
                    os.path.join(fake_root, method, "masks")) else None
                if mask_path and not os.path.isfile(mask_path):
                    mask_path = None
                self.data.append((img_path, mask_path, 1))

        # Apply size limit by random sampling
        if self.size_limit is not None and len(self.data) > self.size_limit:
            self.data = random.sample(self.data, self.size_limit)

        if len(self.data) == 0:
            raise RuntimeError(f"❌ No images found for '{subset}' in '{root_dir}' (real or fake).")
        else:
            print(f"✅ [{subset.upper()}] Loaded {len(self.data)} samples from '{root_dir}'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")

        # Apply JPEG compression
        image = self.apply_jpeg_compression(image)
        # Apply platform-specific color shifts
        image = self.apply_platform_artifacts(image)

        if self.transform:
            image = self.transform(image)

        if mask_path:
            mask = Image.open(mask_path).convert("L")
            mask = mask.resize(Config.INPUT_SIZE, resample=Image.NEAREST)
            mask = torch.from_numpy(np.array(mask) / 255.0).float().unsqueeze(0)
        else:
            mask = torch.zeros(1, *Config.INPUT_SIZE, dtype=torch.float)

        return {"image": image, "mask": mask, "label": label, "path": img_path}

    def apply_jpeg_compression(self, image):
        quality = np.random.randint(50, 91)
        image_np = np.array(image)
        _, compressed = cv2.imencode('.jpg', image_np, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        decompressed = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
        return Image.fromarray(decompressed)

    def apply_platform_artifacts(self, image):
        if np.random.random() < 0.5:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(np.random.uniform(0.8, 1.2))
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(np.random.uniform(0.9, 1.1))
        return image


def get_dataloader(dataset_name, subset, batch_size=None, shuffle=True):
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    data_root = os.path.join(Config.DATA_ROOT, dataset_name)

    transform = transforms.Compose([
        transforms.Resize(Config.INPUT_SIZE),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ])

    # Define size limits for each dataset and subset
    size_limits = {
        'celebahq': {
            'train': 980,  # Max 84,020
            'valid': 10,   # Max 8,520
            'test': 10    # Max 12,972
        },
        'ffhq': {
            'train': 980,  # Max 49,000
            'valid': 20,   # Max 5,000
            'test': 0        # No test split
        }
    }

    dataset = DeepfakeDataset(
        data_root,
        subset,
        transform=transform,
        use_temporal=Config.DATASET_CONFIGS.get(dataset_name, {}).get("use_temporal", False),
        train_size=size_limits.get(dataset_name, {}).get('train', None),
        val_size=size_limits.get(dataset_name, {}).get('valid', None),
        test_size=size_limits.get(dataset_name, {}).get('test', None),
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    return loader