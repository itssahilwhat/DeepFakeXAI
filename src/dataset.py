# src/dataset.py

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.config import Config
import random
import pandas as pd

class DeepfakeDataset(Dataset):
    def __init__(self, split, img_size=Config.IMG_SIZE, aug_strong=True, manifest_path=None):
        self.split = split
        self.img_size = img_size
        self.aug_strong = aug_strong
        self.manifest_path = manifest_path or os.path.join(Config.DATA_ROOT, 'manifest.csv')
        self.samples = self._load_manifest()
        self.transform = self._build_transform()

    def _load_manifest(self):
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Manifest file not found at {self.manifest_path}. Run generate_manifest.py.")
        df = pd.read_csv(self.manifest_path)
        return df[df['split'] == self.split].to_dict('records')

    def _build_transform(self):
        if self.aug_strong and self.split == 'train':
            # --- FINAL, DEFINITIVE AUGMENTATION PIPELINE ---
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(-0.05, 0.05),
                    rotate=(-15, 15),
                    p=0.5
                ),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GridDropout(ratio=0.1, unit_size_range=(25, 26), p=0.5),
                A.Blur(blur_limit=3, p=0.2),
                A.Normalize(),
                ToTensorV2(),
            ])
        else:
            return A.Compose([A.Normalize(), ToTensorV2()])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            img = cv2.imread(sample['img_path'])
            if img is None: raise IOError(f"cv2.imread returned None for {sample['img_path']}")
        except Exception as e:
            print(f"[ERROR] Failed to load image: {e}. Returning random sample.")
            return self.__getitem__(random.randint(0, len(self.samples) - 1))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        mask_path = sample.get('mask_path')
        if pd.notna(mask_path) and mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, 0)
            if mask is None:
                mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            else:
                mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
                mask = (mask > 127).astype(np.float32)
        else:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        transformed = self.transform(image=img, mask=mask)
        img_tensor = transformed['image']
        mask_tensor = transformed['mask'].unsqueeze(0)
        label = torch.tensor(sample['label'], dtype=torch.long)

        return img_tensor, mask_tensor, label

    @property
    def labels(self):
        return [s['label'] for s in self.samples]

    def get_sample_weights(self):
        from collections import Counter
        labels = self.labels
        counts = Counter(labels)
        weights = [1.0 / counts[label] for label in labels]
        return torch.DoubleTensor(weights)


def get_dataloader(split, batch_size, **kwargs):
    dataset = DeepfakeDataset(split=split, aug_strong=(split == 'train'))

    sampler = None
    shuffle = (split == 'train')
    if split == 'train':
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=dataset.get_sample_weights(),
            num_samples=len(dataset),
            replacement=True
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )