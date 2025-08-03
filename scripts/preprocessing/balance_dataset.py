#!/usr/bin/env python3

"""
Dataset balancing script to create perfect 50-50% balance and exact 80-10-10 splits.
This script will create a balanced subset of the original dataset.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.append('.')

def balance_dataset():
    print("Creating Perfectly Balanced Dataset")
    print("="*50)
    
    manifest_path = Path('data/wacv_data/manifest.csv')
    if not manifest_path.exists():
        print("Original manifest not found!")
        return False
    
    df = pd.read_csv(manifest_path)
    print(f"Original dataset: {len(df):,} samples")
    
    real_samples = df[df['label'] == 0]
    fake_samples = df[df['label'] == 1]
    
    print(f"  Real samples: {len(real_samples):,}")
    print(f"  Fake samples: {len(fake_samples):,}")
    
    max_balanced_size = min(len(real_samples), len(fake_samples)) * 2
    samples_per_class = max_balanced_size // 2
    
    samples_per_class = (samples_per_class // 10) * 10
    max_balanced_size = samples_per_class * 2
    
    print(f"\nTarget balanced dataset: {max_balanced_size:,} samples")
    print(f"  Real samples: {samples_per_class:,}")
    print(f"  Fake samples: {samples_per_class:,}")
    
    real_balanced = real_samples.sample(n=samples_per_class, random_state=42)
    fake_balanced = fake_samples.sample(n=samples_per_class, random_state=42)
    
    balanced_df = pd.concat([real_balanced, fake_balanced], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nCreated balanced dataset: {len(balanced_df):,} samples")
    print(f"  Real: {len(balanced_df[balanced_df['label'] == 0]):,} ({len(balanced_df[balanced_df['label'] == 0])/len(balanced_df)*100:.1f}%)")
    print(f"  Fake: {len(balanced_df[balanced_df['label'] == 1]):,} ({len(balanced_df[balanced_df['label'] == 1])/len(balanced_df)*100:.1f}%)")
    
    total_samples = len(balanced_df)
    samples_per_split = total_samples // 10
    
    train_size = samples_per_split * 8
    val_size = samples_per_split * 1
    test_size = samples_per_split * 1
    
    print(f"\nCreating exact 80-10-10 splits:")
    print(f"  Train: {train_size:,} samples ({train_size/total_samples*100:.1f}%)")
    print(f"  Valid: {val_size:,} samples ({val_size/total_samples*100:.1f}%)")
    print(f"  Test:  {test_size:,} samples ({test_size/total_samples*100:.1f}%)")
    
    real_samples_list = balanced_df[balanced_df['label'] == 0].reset_index(drop=True)
    fake_samples_list = balanced_df[balanced_df['label'] == 1].reset_index(drop=True)
    
    real_per_split = len(real_samples_list) // 10
    fake_per_split = len(fake_samples_list) // 10
    
    train_real = real_samples_list[:real_per_split * 8]
    val_real = real_samples_list[real_per_split * 8:real_per_split * 9]
    test_real = real_samples_list[real_per_split * 9:]
    
    train_fake = fake_samples_list[:fake_per_split * 8]
    val_fake = fake_samples_list[fake_per_split * 8:fake_per_split * 9]
    test_fake = fake_samples_list[fake_per_split * 9:]
    
    train_df = pd.concat([train_real, train_fake], ignore_index=True)
    val_df = pd.concat([val_real, val_fake], ignore_index=True)
    test_df = pd.concat([test_real, test_fake], ignore_index=True)
    
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_df.loc[:, 'split'] = 'train'
    val_df.loc[:, 'split'] = 'valid'
    test_df.loc[:, 'split'] = 'test'
    
    final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    print(f"\nVerifying balance in each split:")
    for split_name, split_df in [('TRAIN', train_df), ('VALID', val_df), ('TEST', test_df)]:
        real_count = len(split_df[split_df['label'] == 0])
        fake_count = len(split_df[split_df['label'] == 1])
        total = len(split_df)
        
        print(f"  {split_name}:")
        print(f"    Real: {real_count:,} ({real_count/total*100:.1f}%)")
        print(f"    Fake: {fake_count:,} ({fake_count/total*100:.1f}%)")
        print(f"    Balance: {'OK' if real_count == fake_count else 'FAIL'}")
    
    balanced_manifest_path = Path('data/wacv_data/manifest_balanced.csv')
    final_df.to_csv(balanced_manifest_path, index=False)
    
    print(f"\nSaved balanced manifest: {balanced_manifest_path}")
    
    original_backup = Path('data/wacv_data/manifest_original.csv')
    df.to_csv(original_backup, index=False)
    print(f"Backed up original manifest: {original_backup}")
    
    final_df.to_csv(manifest_path, index=False)
    
    print(f"\nDataset balancing completed!")
    print(f"  Original: {len(df):,} samples (imbalanced)")
    print(f"  Balanced: {len(final_df):,} samples (50-50% perfect)")
    print(f"  Perfect 80-10-10 splits achieved!")
    
    return True

if __name__ == "__main__":
    balance_dataset() 