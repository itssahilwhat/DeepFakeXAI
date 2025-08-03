#!/usr/bin/env python3

"""
Quick dataset verification script.
This script quickly checks if your dataset is properly balanced and split.
"""

import pandas as pd
from pathlib import Path

def verify_dataset():
    """Quick verification of dataset balance and splits"""
    print("Quick Dataset Verification")
    print("="*40)
    
    # Check if manifest exists
    manifest_path = Path('data/wacv_data/manifest.csv')
    if not manifest_path.exists():
        print("Dataset manifest not found!")
        print("Please run: python scripts/preprocessing/build_manifest.py")
        return False
    
    # Load manifest
    df = pd.read_csv(manifest_path)
    print(f"Loaded manifest with {len(df):,} samples")
    
    # Overall balance
    total_real = len(df[df['label'] == 0])
    total_fake = len(df[df['label'] == 1])
    total = len(df)
    
    print(f"\nOverall Balance:")
    print(f"  Real: {total_real:,} ({total_real/total*100:.1f}%)")
    print(f"  Fake: {total_fake:,} ({total_fake/total*100:.1f}%)")
    
    balanced_overall = 0.4 <= total_real/total <= 0.6
    print(f"  Overall Balanced: {'OK' if balanced_overall else 'FAIL'}")
    
    # Split ratios
    train_count = len(df[df['split'] == 'train'])
    valid_count = len(df[df['split'] == 'valid'])
    test_count = len(df[df['split'] == 'test'])
    
    train_ratio = train_count / total
    valid_ratio = valid_count / total
    test_ratio = test_count / total
    
    print(f"\nSplit Ratios:")
    print(f"  Train: {train_ratio*100:.1f}% (target: 80%) {'OK' if 0.75 <= train_ratio <= 0.85 else 'FAIL'}")
    print(f"  Valid: {valid_ratio*100:.1f}% (target: 10%) {'OK' if 0.08 <= valid_ratio <= 0.12 else 'FAIL'}")
    print(f"  Test:  {test_ratio*100:.1f}% (target: 10%) {'OK' if 0.08 <= test_ratio <= 0.12 else 'FAIL'}")
    
    correct_splits = all([
        0.75 <= train_ratio <= 0.85,
        0.08 <= valid_ratio <= 0.12,
        0.08 <= test_ratio <= 0.12
    ])
    
    # Per-split balance
    print(f"\nPer-Split Balance:")
    balanced_splits = True
    
    for split in ['train', 'valid', 'test']:
        split_df = df[df['split'] == split]
        split_real = len(split_df[split_df['label'] == 0])
        split_fake = len(split_df[split_df['label'] == 1])
        split_total = len(split_df)
        
        real_ratio = split_real / split_total
        fake_ratio = split_fake / split_total
        balanced = 0.4 <= real_ratio <= 0.6 and 0.4 <= fake_ratio <= 0.6
        
        print(f"  {split.upper()}: Real {real_ratio*100:.1f}% / Fake {fake_ratio*100:.1f}% {'OK' if balanced else 'FAIL'}")
        
        if not balanced:
            balanced_splits = False
    
    # Final assessment
    print(f"\nFinal Assessment:")
    if balanced_overall and correct_splits and balanced_splits:
        print("  Dataset is PERFECTLY balanced and split!")
        print("  Ready for training!")
        return True
    else:
        print("  Dataset needs adjustment:")
        if not balanced_overall:
            print("    - Overall balance is not 40-60%")
        if not correct_splits:
            print("    - Split ratios are not 80-10-10")
        if not balanced_splits:
            print("    - Some splits are not balanced")
        print("\nTips:")
        print("  - Ensure equal number of real/fake images in each split")
        print("  - Use 80% for train, 10% for valid, 10% for test")
        print("  - Run: python scripts/analyze.py for detailed analysis")
        return False

if __name__ == "__main__":
    verify_dataset() 