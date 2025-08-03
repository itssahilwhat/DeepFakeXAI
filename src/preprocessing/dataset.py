import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
from collections import Counter
import random

class DeepfakeDataset(Dataset):
    def __init__(self, split, config, aug_strong=True, manifest_path=None, supervision_filter=None):
        self.split = split
        self.config = config
        self.img_size = config.IMG_SIZE
        self.aug_strong = aug_strong
        self.supervision_filter = supervision_filter  # Filter by supervision type (A, B, C)
        self.manifest_path = manifest_path or getattr(config, 'MANIFEST_PATH', None) or os.path.join(config.DATA_ROOT, 'manifest.csv')
        self.samples = self._load_manifest()
        self.transform = self._build_transform()
        
        self.class_weights = self._calculate_class_weights()
        
        if config.USE_STRATIFIED_SAMPLING and split == 'train':
            self.stratified_indices = self._get_stratified_indices()
        
        print(f"Dataset loaded ({split}): {len(self.samples)} samples")
        if self.supervision_filter:
            print(f"Filtered by supervision: {self.supervision_filter}")

    def _load_manifest(self):
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Manifest file not found at {self.manifest_path}")
        df = pd.read_csv(self.manifest_path)
        if df.empty:
            raise ValueError(f"Manifest file {self.manifest_path} is empty")
        
        # Filter by split
        df = df[df['split'] == self.split]
        
        # Filter by supervision type if specified
        if self.supervision_filter:
            df = df[df['supervision'] == self.supervision_filter]
        
        return df.to_dict('records')

    def _calculate_class_weights(self):
        labels = [s['label'] for s in self.samples]
        class_counts = Counter(labels)
        total_samples = len(labels)
        
        weights = {cls: total_samples / (len(class_counts) * count) 
                  for cls, count in class_counts.items()}
        
        return weights

    def _get_stratified_indices(self):
        labels = [s['label'] for s in self.samples]
        label_indices = {}
        
        for idx, label in enumerate(labels):
            if label not in label_indices:
                label_indices[label] = []
            label_indices[label].append(idx)
        
        min_class_count = min(len(indices) for indices in label_indices.values())
        balanced_indices = []
        
        for label, indices in label_indices.items():
            if len(indices) > min_class_count:
                balanced_indices.extend(random.choices(indices, k=min_class_count))
            else:
                balanced_indices.extend(indices)
        
        random.shuffle(balanced_indices)
        return balanced_indices

    def _build_transform(self):
        if self.aug_strong and self.split == 'train':
            # Common transforms for all images
            common_transforms = A.Compose([
                A.Resize(height=self.img_size, width=self.img_size),
                A.Normalize(),
                ToTensorV2(),
            ], is_check_shapes=False)
            
            # Strong augmentation for real images only (updated for albumentations 2.0.8)
            real_augment = A.Compose([
                A.Resize(height=self.img_size, width=self.img_size),
                A.HorizontalFlip(p=0.5),
                A.Affine(scale=(0.9, 1.1), translate_percent=0.1, rotate=(-15, 15), p=0.5),
                
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
                
                A.GaussNoise(std_range=(0.1, 0.3), p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.2),
                A.MotionBlur(blur_limit=7, p=0.2),
                
                A.CoarseDropout(num_holes_range=(4, 8), hole_height_range=(0.1, 0.15), hole_width_range=(0.1, 0.15), p=0.3),
                
                A.Normalize(),
                ToTensorV2(),
            ], is_check_shapes=False)
            
            return common_transforms, real_augment
        else:
            return A.Compose([
                A.Resize(height=self.img_size, width=self.img_size),
                A.Normalize(),
                ToTensorV2(),
            ], is_check_shapes=False)

    def _preprocess_mask(self, mask_path, supervision_type):
        """Preprocess mask based on supervision type"""
        # Handle NaN values from pandas
        if pd.isna(mask_path) or not mask_path or not os.path.exists(str(mask_path)):
            return np.zeros((self.img_size, self.img_size), dtype=np.float32)
        
        try:
            mask = cv2.imread(str(mask_path), 0)
            if mask is None:
                return np.zeros((self.img_size, self.img_size), dtype=np.float32)
            
            # Resize to target size
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            
            # Normalize to [0, 1]
            mask = (mask > 127).astype(np.float32)
            
            # Apply supervision-specific preprocessing
            if supervision_type == 'B':
                # Weak supervision: apply random blur/erosion to simulate noisy masks
                if random.random() < 0.3:
                    kernel_size = random.choice([3, 5, 7])
                    mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
                if random.random() < 0.2:
                    kernel = np.ones((3, 3), np.uint8)
                    mask = cv2.erode(mask, kernel, iterations=1)
            
            return mask
            
        except Exception as e:
            print(f"Error loading mask {mask_path}: {e}")
            return np.zeros((self.img_size, self.img_size), dtype=np.float32)

    def _apply_mixup(self, img1, img2, label1, label2):
        alpha = self.config.MIXUP_ALPHA
        lam = np.random.beta(alpha, alpha)
        
        mixed_img = lam * img1 + (1 - lam) * img2
        mixed_label = lam * label1 + (1 - lam) * label2
        
        return mixed_img, mixed_label

    def _apply_cutmix(self, img1, img2, label1, label2):
        alpha = self.config.CUTMIX_ALPHA
        lam = np.random.beta(alpha, alpha)
        
        W, H = self.img_size, self.img_size
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        mixed_img = img1.clone()
        mixed_img[:, bby1:bby2, bbx1:bbx2] = img2[:, bby1:bby2, bbx1:bbx2]
        
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        mixed_label = lam * label1 + (1 - lam) * label2
        
        return mixed_img, mixed_label

    def __len__(self):
        if hasattr(self, 'stratified_indices'):
            return len(self.stratified_indices)
        return len(self.samples)

    def __getitem__(self, idx):
        if hasattr(self, 'stratified_indices'):
            if idx >= len(self.stratified_indices):
                idx = idx % len(self.stratified_indices)
            idx = self.stratified_indices[idx]
        
        sample = self.samples[idx]
        img_path = sample['img_path']
        supervision_type = sample.get('supervision', 'A')
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise IOError(f"Failed to load image at {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load and preprocess mask based on supervision type
        mask_path = sample.get('mask_path', '')
        mask = self._preprocess_mask(mask_path, supervision_type)
        
        # Apply transformations based on label
        # Force label to be 0 or 1 (int) to fix weighted sampling
        raw_label = sample['label']
        if isinstance(raw_label, (float, str)):
            label = int(float(raw_label) > 0.5)  # threshold at 0.5
        else:
            label = int(raw_label)
        label = torch.tensor(label, dtype=torch.long)  # Use long for classification
        
        if self.split == 'train' and self.aug_strong:
            # Use strong augmentation for real images, common for fake
            if label == 0:  # real image
                transformed = self.transform[1](image=img, mask=mask)  # real_augment
            else:  # fake image
                transformed = self.transform[0](image=img, mask=mask)  # common_transforms
        else:
            # Use common transforms for validation/test
            transformed = self.transform(image=img, mask=mask)
        
        img_tensor = transformed['image']
        mask_tensor = transformed['mask']
        
        # Apply mixup/cutmix for training
        if (self.split == 'train' and self.config.USE_MIXUP and 
            random.random() < 0.5):
            other_idx = random.randint(0, len(self.samples) - 1)
            other_sample = self.samples[other_idx]
            other_img = cv2.imread(other_sample['img_path'])
            other_img = cv2.cvtColor(other_img, cv2.COLOR_BGR2RGB)
            other_mask = self._preprocess_mask(other_sample.get('mask_path', ''), 
                                             other_sample.get('supervision', 'A'))
            # Force other label to be 0 or 1 (int) as well
            other_raw_label = other_sample['label']
            if isinstance(other_raw_label, (float, str)):
                other_label = int(float(other_raw_label) > 0.5)  # threshold at 0.5
            else:
                other_label = int(other_raw_label)
            other_label = torch.tensor(other_label, dtype=torch.long)
            
            # Apply same transform logic for other sample
            if self.split == 'train' and self.aug_strong:
                if other_label == 0:  # real image
                    other_transformed = self.transform[1](image=other_img, mask=other_mask)
                else:  # fake image
                    other_transformed = self.transform[0](image=other_img, mask=other_mask)
            else:
                other_transformed = self.transform(image=other_img, mask=other_mask)
            
            other_img_tensor = other_transformed['image']
            
            if self.config.USE_MIXUP and random.random() < 0.5:
                img_tensor, mixed_label = self._apply_mixup(img_tensor, other_img_tensor, label, other_label)
                # Convert mixed label back to long for classification
                label = torch.tensor(int(mixed_label.item() > 0.5), dtype=torch.long)
            elif self.config.USE_CUTMIX:
                img_tensor, mixed_label = self._apply_cutmix(img_tensor, other_img_tensor, label, other_label)
                # Convert mixed label back to long for classification
                label = torch.tensor(int(mixed_label.item() > 0.5), dtype=torch.long)
        
        return img_tensor, mask_tensor, label

    @property
    def labels(self):
        # Force all labels to be 0 or 1
        labels = []
        for s in self.samples:
            raw_label = s['label']
            if isinstance(raw_label, (float, str)):
                label = int(float(raw_label) > 0.5)  # threshold at 0.5
            else:
                label = int(raw_label)
            labels.append(label)
        return labels

    def get_sample_weights(self):
        """Compute inverse frequency weights for balanced sampling"""
        # Force all labels to be 0 or 1 for proper sampling
        labels = []
        for s in self.samples:
            raw_label = s['label']
            if isinstance(raw_label, (float, str)):
                label = int(float(raw_label) > 0.5)  # threshold at 0.5
            else:
                label = int(raw_label)
            labels.append(label)
        
        num_real = labels.count(0)
        num_fake = labels.count(1)
        
        # Since dataset is perfectly balanced (50-50%), use equal weights
        # This ensures we get true random sampling without bias
        weight_real = 1.0
        weight_fake = 1.0
        
        # Assign weights based on labels
        sample_weights = [weight_real if label == 0 else weight_fake for label in labels]
        
        print(f"Sampling weights - Real: {weight_real:.6f}, Fake: {weight_fake:.6f}")
        print(f"Expected ratio: {weight_real/weight_fake:.3f} (real/fake)")
        
        return torch.DoubleTensor(sample_weights)

def get_dataloader(split, config, batch_size_override=None, supervision_filter=None):
    dataset = DeepfakeDataset(split, config, supervision_filter=supervision_filter)
    batch_size = batch_size_override or config.BATCH_SIZE
    
    # Use simple random sampling since dataset is perfectly balanced
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=(split == 'train')
    )
    
    return dataloader
