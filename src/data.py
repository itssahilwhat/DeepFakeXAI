# src/data.py
import os
import cv2
import random
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms
from config import Config


class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, subset, transform=None, use_temporal=False,
                 train_size=None, val_size=None, test_size=None):
        self.root_dir = root_dir
        self.subset = subset
        self.transform = transform
        self.use_temporal = use_temporal
        self.data = []
        self.dataset_name = os.path.basename(root_dir)  # 'celebahq' or 'ffhq'

        # Set size limit based on subset
        self.size_limit = {
            'train': train_size or Config.TRAIN_SIZE,
            'valid': val_size or Config.VAL_SIZE,
            'test': test_size or Config.TEST_SIZE
        }.get(subset, None)

        if not os.path.isdir(root_dir):
            raise RuntimeError(f"❌ Dataset root not found: {root_dir}")

        cfg = Config.DATASET_CONFIGS.get(self.dataset_name, {})
        has_masks = cfg.get("has_masks", True)

        # ====== Load real images ======
        real_dir = os.path.join(root_dir, "real", subset)
        if os.path.isdir(real_dir):
            real_images = [f for f in os.listdir(real_dir)
                           if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            for fname in real_images:
                img_path = os.path.join(real_dir, fname)
                self.data.append((img_path, None, 0))

        # ====== Load fake images ======
        fake_root = os.path.join(root_dir, "fake")
        if os.path.isdir(fake_root):
            for method in os.listdir(fake_root):
                method_dir = os.path.join(fake_root, method)

                # Handle different fake directory structures
                image_dirs = []
                mask_dirs = []

                # Structure 1: method/images/subset and method/masks/subset
                img_dir1 = os.path.join(method_dir, "images", subset)
                mask_dir1 = os.path.join(method_dir, "masks", subset)

                # Structure 2: method/subset directly (without images/masks)
                img_dir2 = os.path.join(method_dir, subset)

                # Structure 3: method/train, method/valid directly (FFHQ style)
                img_dir3 = method_dir if subset in os.listdir(method_dir) else None

                # Check which structure exists
                if os.path.isdir(img_dir1):
                    image_dirs.append(img_dir1)
                    mask_dirs.append(mask_dir1)
                elif os.path.isdir(img_dir2):
                    image_dirs.append(img_dir2)
                    mask_dirs.append(None)  # No masks for this structure
                elif img_dir3 and os.path.isdir(os.path.join(method_dir, subset)):
                    image_dirs.append(os.path.join(method_dir, subset))
                    mask_dirs.append(None)

                # Special handling for FFHQ p2 structure
                if self.dataset_name == "ffhq" and method == "p2":
                    if os.path.isdir(os.path.join(method_dir, "train")) and subset == "train":
                        image_dirs.append(os.path.join(method_dir, "train"))
                        mask_dirs.append(None)
                    elif os.path.isdir(os.path.join(method_dir, "valid")) and subset == "valid":
                        image_dirs.append(os.path.join(method_dir, "valid"))
                        mask_dirs.append(None)

                # Process all found image directories
                for img_dir, mask_dir in zip(image_dirs, mask_dirs):
                    if os.path.isdir(img_dir):
                        method_images = [f for f in os.listdir(img_dir)
                                         if f.lower().endswith((".jpg", ".jpeg", ".png"))]

                        # Sample every 3rd frame for temporal data
                        if self.use_temporal:
                            method_images = method_images[::3]

                        # Get mask directory if exists
                        final_mask_dir = None
                        if mask_dir and os.path.isdir(mask_dir):
                            final_mask_dir = mask_dir
                        elif has_masks:
                            # Try to find masks in alternative location
                            possible_mask_dir = img_dir.replace("images", "masks")
                            if os.path.isdir(possible_mask_dir):
                                final_mask_dir = possible_mask_dir

                        for fname in method_images:
                            img_path = os.path.join(img_dir, fname)
                            mask_path = os.path.join(final_mask_dir, fname) if final_mask_dir else None

                            # Verify mask exists if expected
                            if has_masks and mask_path and not os.path.isfile(mask_path):
                                mask_path = None

                            self.data.append((img_path, mask_path, 1))

        # Apply size limit by random sampling
        if self.size_limit is not None and len(self.data) > self.size_limit:
            self.data = random.sample(self.data, self.size_limit)

        if len(self.data) == 0:
            raise RuntimeError(f"❌ No images found for '{subset}' in '{root_dir}'")
        else:
            print(f"✅ [{subset.upper()}] Loaded {len(self.data)} samples from '{root_dir}'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.data[idx]

        # Use OpenCV for faster image loading
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)  # Only convert if transforms need PIL

        if not (Config.PRESERVE_ORIGINAL and self.subset != 'train'):
            # Apply JPEG compression
            image = self.apply_jpeg_compression(image)
            # Apply platform-specific color shifts
            image = self.apply_platform_artifacts(image)
            # Apply diffusion artifacts if enabled
            if Config.USE_DIFFUSION_AUG and self.subset == 'train':
                image = self.apply_diffusion_artifacts(image)

        if self.transform:
            image = self.transform(image)

        # Process mask
        if mask_path and os.path.exists(mask_path):
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

    def apply_diffusion_artifacts(self, image):
        """Simulate diffusion model artifacts"""
        if random.random() < 0.3:  # 30% chance to apply
            artifact_type = random.choice(["blur", "color_shift", "texture"])

            if artifact_type == "blur":
                kernel_size = random.choice([3, 5, 7])
                return image.filter(ImageFilter.GaussianBlur(radius=kernel_size / 2))

            elif artifact_type == "color_shift":
                enhancer = ImageEnhance.Color(image)
                return enhancer.enhance(random.uniform(0.8, 1.2))

            elif artifact_type == "texture":
                np_img = np.array(image)
                noise = np.random.normal(0, 15, np_img.shape).astype(np.uint8)
                blended = cv2.addWeighted(np_img, 0.9, noise, 0.1, 0)
                return Image.fromarray(blended)

        return image


def get_dataloader(dataset_name, subset, batch_size=None, shuffle=True):
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    data_root = os.path.join(Config.DATA_ROOT, dataset_name)
    if Config.PRESERVE_ORIGINAL and subset != 'train':
        transform = transforms.Compose([
            transforms.Resize(Config.INPUT_SIZE),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(Config.INPUT_SIZE),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
        ])

    dataset = DeepfakeDataset(
        data_root,
        subset,
        transform=transform,
        use_temporal=Config.DATASET_CONFIGS.get(dataset_name, {}).get("use_temporal", False),
        train_size=Config.TRAIN_SIZE if subset == 'train' else None,
        val_size=Config.VAL_SIZE if subset == 'valid' else None,
        test_size=Config.TEST_SIZE if subset == 'test' else None,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=min(12, os.cpu_count()),  # Safer worker count
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        drop_last=True
    )
    return loader