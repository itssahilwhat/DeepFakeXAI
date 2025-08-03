#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import cv2
from pathlib import Path

def assess_mask_quality(mask_path):
    """Assess mask quality to determine supervision type"""
    if not mask_path or not os.path.exists(mask_path):
        return 'C'  # No mask available
    
    try:
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            return 'C'
        
        # Check if mask is mostly empty (global fake)
        mask_area = np.sum(mask > 127)
        total_pixels = mask.shape[0] * mask.shape[1]
        coverage = mask_area / total_pixels
        
        if coverage < 0.01:  # Less than 1% coverage
            return 'C'  # Global fake, no local mask
        elif coverage > 0.8:  # More than 80% coverage
            return 'B'  # Likely noisy/partial mask
        else:
            return 'A'  # Good quality mask
    except:
        return 'C'

def build_dataset_manifest():
    """Build dataset manifest with supervision types based on mask quality"""
    print("Analyzing dataset structure...")
    
    data_root = Path('data/wacv_data')
    samples = []
    
    # Process each domain
    for domain in ['celebahq', 'ffhq', 'celeba']:
        domain_path = data_root / domain
        
        # Process real images (no masks, supervision C)
        if domain == 'celeba':
            # Special handling for celeba dataset
            celeba_images_path = domain_path / 'img_align_celeba'
            celeba_partition_path = domain_path / 'list_eval_partition.csv'
            
            if celeba_images_path.exists() and celeba_partition_path.exists():
                # Load partition information
                partition_df = pd.read_csv(celeba_partition_path)
                partition_df['image_id'] = partition_df['image_id'].astype(str)
                
                # Map partition to split
                partition_to_split = {0: 'train', 1: 'valid', 2: 'test'}
                
                for _, row in partition_df.iterrows():
                    img_file = celeba_images_path / row['image_id']
                    if img_file.exists():
                        split = partition_to_split[row['partition']]
                        samples.append({
                            'img_path': str(img_file),
                            'mask_path': '',
                            'label': 0,  # All celeba images are real
                            'split': split,
                            'supervision': 'A',  # Real images get full supervision
                            'domain': domain,
                            'method': 'celeba_real'
                        })
        else:
            # Standard handling for celebahq and ffhq
            real_path = domain_path / 'real'
            if real_path.exists():
                for split in ['train', 'valid', 'test']:
                    split_path = real_path / split
                    if split_path.exists():
                        for img_file in split_path.glob('*.png'):
                            samples.append({
                                'img_path': str(img_file),
                                'mask_path': '',
                                'label': 0,
                                'split': split,
                                'supervision': 'A',  # Real images get full supervision
                                'domain': domain
                            })
        
        # Process fake images with masks (skip for celeba as it's all real)
        if domain != 'celeba':
            fake_path = domain_path / 'fake'
            if fake_path.exists():
                for method in fake_path.iterdir():
                    if method.is_dir():
                        method_name = method.name
                        
                        # Process images
                        images_path = method / 'images'
                        masks_path = method / 'masks'
                        
                        if images_path.exists():
                            for split in ['train', 'valid', 'test']:
                                split_img_path = images_path / split
                                split_mask_path = masks_path / split if masks_path.exists() else None
                                
                                if split_img_path.exists():
                                    for img_file in split_img_path.glob('*.png'):
                                        # Find corresponding mask
                                        mask_path = ''
                                        if split_mask_path and split_mask_path.exists():
                                            mask_file = split_mask_path / f"{img_file.stem}.png"
                                            if mask_file.exists():
                                                mask_path = str(mask_file)
                                        
                                        # Determine supervision type
                                        supervision = assess_mask_quality(mask_path)
                                        
                                        samples.append({
                                            'img_path': str(img_file),
                                            'mask_path': mask_path,
                                            'label': 1,
                                            'split': split,
                                            'supervision': supervision,
                                            'domain': domain,
                                            'method': method_name
                                        })
    
    # Convert to DataFrame
    df = pd.DataFrame(samples)
    
    print(f"Dataset Statistics:")
    print(f"Total samples: {len(df):,}")
    print(f"Real samples: {len(df[df['label'] == 0]):,}")
    print(f"Fake samples: {len(df[df['label'] == 1]):,}")
    
    print(f"Supervision Distribution:")
    supervision_counts = df['supervision'].value_counts()
    for supervision, count in supervision_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {supervision}: {count:,} ({percentage:.1f}%)")
    
    print(f"Domain Distribution:")
    domain_counts = df['domain'].value_counts()
    for domain, count in domain_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {domain}: {count:,} ({percentage:.1f}%)")
    
    print(f"Method Distribution (Fake only):")
    fake_df = df[df['label'] == 1]
    if 'method' in fake_df.columns:
        method_counts = fake_df['method'].value_counts()
        for method, count in method_counts.items():
            percentage = (count / len(fake_df)) * 100
            print(f"  {method}: {count:,} ({percentage:.1f}%)")
    
    print(f"Split Distribution:")
    split_counts = df['split'].value_counts()
    for split, count in split_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {split}: {count:,} ({percentage:.1f}%)")
    
    # Verify balance and splits
    print(f"\n🔍 Balance and Split Verification:")
    
    # Check overall balance
    total_real = len(df[df['label'] == 0])
    total_fake = len(df[df['label'] == 1])
    total = len(df)
    
    print(f"Overall Balance:")
    print(f"  Real: {total_real:,} ({total_real/total*100:.1f}%)")
    print(f"  Fake: {total_fake:,} ({total_fake/total*100:.1f}%)")
    print(f"  Balanced: {'✅' if 0.4 <= total_real/total <= 0.6 else '❌'}")
    
    # Check split ratios
    train_count = len(df[df['split'] == 'train'])
    valid_count = len(df[df['split'] == 'valid'])
    test_count = len(df[df['split'] == 'test'])
    
    train_ratio = train_count / total
    valid_ratio = valid_count / total
    test_ratio = test_count / total
    
    print(f"Split Ratios:")
    print(f"  Train: {train_ratio*100:.1f}% (target: 80%) {'✅' if 0.75 <= train_ratio <= 0.85 else '❌'}")
    print(f"  Valid: {valid_ratio*100:.1f}% (target: 10%) {'✅' if 0.08 <= valid_ratio <= 0.12 else '❌'}")
    print(f"  Test:  {test_ratio*100:.1f}% (target: 10%) {'✅' if 0.08 <= test_ratio <= 0.12 else '❌'}")
    
    # Check per-split balance
    print(f"Per-Split Balance:")
    for split in ['train', 'valid', 'test']:
        split_df = df[df['split'] == split]
        split_real = len(split_df[split_df['label'] == 0])
        split_fake = len(split_df[split_df['label'] == 1])
        split_total = len(split_df)
        
        real_ratio = split_real / split_total
        fake_ratio = split_fake / split_total
        balanced = 0.4 <= real_ratio <= 0.6 and 0.4 <= fake_ratio <= 0.6
        
        print(f"  {split.upper()}: Real {real_ratio*100:.1f}% / Fake {fake_ratio*100:.1f}% {'✅' if balanced else '❌'}")
    
    # Save manifest
    output_path = data_root / 'manifest.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\nManifest saved: {output_path}")
    print(f"Columns: {list(df.columns)}")
    
    return str(output_path)

if __name__ == '__main__':
    build_dataset_manifest() 