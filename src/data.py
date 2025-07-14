import os
import cv2
import random
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms
from src.config import Config


class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, subset, transform=None, use_temporal=False,
                 train_size=None, val_size=None, test_size=None):
        self.root_dir = root_dir
        self.subset = subset
        self.transform = transform
        self.use_temporal = use_temporal
        self.data = []
        self.dataset_name = os.path.basename(root_dir)

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
                    mask_dirs.append(None)
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
                            possible_mask_dir = img_dir.replace("images", "masks")
                            if os.path.isdir(possible_mask_dir):
                                final_mask_dir = possible_mask_dir

                        for fname in method_images:
                            img_path = os.path.join(img_dir, fname)
                            mask_path = os.path.join(final_mask_dir, fname) if final_mask_dir else None

                            if has_masks and mask_path and not os.path.isfile(mask_path):
                                mask_path = None

                            self.data.append((img_path, mask_path, 1))

        # Apply size limit by random sampling
        if self.size_limit is not None and len(self.data) > self.size_limit:
            indices = random.sample(range(len(self.data)), self.size_limit)
            self.data = [self.data[i] for i in indices]

        # Control augmentations based on subset
        self.apply_augmentations = True
        if subset == 'test' and not Config.AUGMENT_TEST:
            self.apply_augmentations = False
        elif subset == 'valid' and not Config.AUGMENT_VALID:
            self.apply_augmentations = False
        elif subset == 'train' and not Config.AUGMENT_TRAIN:
            self.apply_augmentations = False

        if len(self.data) == 0:
            raise RuntimeError(f"❌ No images found for '{subset}' in '{root_dir}'")
        else:
            aug_status = "WITH" if self.apply_augmentations else "WITHOUT"
            print(f"✅ [{subset.upper()}] Loaded {len(self.data)} samples from '{root_dir}' ({aug_status} augmentation)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.data[idx]

        # Optimized image loading with OpenCV
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', Config.INPUT_SIZE, (0, 0, 0))

        # Minimal augmentation for speed
        if self.apply_augmentations and self.subset == 'train':
            # Apply minimal augmentations with low frequency for speed
            if np.random.random() < 0.2:  # 20% chance
                image = self.apply_minimal_augmentations(image)

        if self.transform:
            image = self.transform(image)

        # Process mask efficiently
        if mask_path and os.path.exists(mask_path):
            try:
                mask = Image.open(mask_path).convert("L")
                mask = mask.resize(Config.INPUT_SIZE, resample=Image.NEAREST)
                mask = torch.from_numpy(np.array(mask) / 255.0).float().unsqueeze(0)
            except Exception as e:
                print(f"Error loading mask {mask_path}: {e}")
                mask = torch.zeros(1, *Config.INPUT_SIZE, dtype=torch.float)
        else:
            mask = torch.zeros(1, *Config.INPUT_SIZE, dtype=torch.float)

        return {"image": image, "mask": mask, "label": label, "path": img_path}

    def apply_minimal_augmentations(self, image):
        """Minimal augmentation pipeline for speed"""
        # Only basic color augmentations
        if np.random.random() < 0.3:
            # Brightness adjustment
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(np.random.uniform(0.9, 1.1))
        
        if np.random.random() < 0.3:
            # Contrast adjustment
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(np.random.uniform(0.95, 1.05))

        return image


def get_transforms(subset='train'):
    """Optimized transforms for speed"""
    if subset == 'train':
        return transforms.Compose([
            transforms.Resize(Config.INPUT_SIZE),
            transforms.RandomHorizontalFlip(p=0.3),  # Reduced for speed
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(Config.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def get_dataloader(dataset_name, subset, batch_size=None, shuffle=True):
    """Optimized dataloader for speed"""
    data_root = os.path.join(Config.DATA_ROOT, dataset_name)
    
    if not os.path.exists(data_root):
        raise RuntimeError(f"Dataset directory not found: {data_root}")
    
    transform = get_transforms(subset)
    
    dataset = DeepfakeDataset(
        root_dir=data_root,
        subset=subset,
        transform=transform,
        use_temporal=Config.DATASET_CONFIGS.get(dataset_name, {}).get("use_temporal", False)
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size or Config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        persistent_workers=True,  # Keep workers alive for speed
        drop_last=shuffle  # Drop last batch during training for consistent batch sizes
    )