#!/usr/bin/env python3
"""
Test script to verify dataset balancing and show project usage
"""
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_utils import get_dataloader, DeepfakeDataset
from configs.config import manifest_csv

def test_dataset_balance():
    """Test dataset balancing across splits"""
    print("=== DATASET BALANCE ANALYSIS ===")
    
    # Load manifest
    df = pd.read_csv(manifest_csv)
    print(f"Total samples: {len(df)}")
    
    # Check splits
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        if len(split_df) > 0:
            real_count = len(split_df[split_df['label'] == 0])
            fake_count = len(split_df[split_df['label'] == 1])
            print(f"\n{split.upper()} split:")
            print(f"  Real: {real_count}, Fake: {fake_count}")
            print(f"  Balance ratio: {real_count/(real_count+fake_count):.3f} : {fake_count/(real_count+fake_count):.3f}")
    
    # Test balanced sampler
    print("\n=== BALANCED SAMPLER TEST ===")
    try:
        train_loader = get_dataloader('train')
        print("✓ Training dataloader created successfully")
        
        # Check first 100 batches
        batch_count = 0
        real_counts = []
        fake_counts = []
        
        for batch_idx, (x, y, mask) in enumerate(train_loader):
            if batch_idx >= 100:  # Check first 100 batches
                break
            real_count = (y == 0).sum().item()
            fake_count = (y == 1).sum().item()
            real_counts.append(real_count)
            fake_counts.append(fake_count)
            if batch_idx < 5:  # Show first 5 batches
                print(f"Batch {batch_idx}: Real={real_count}, Fake={fake_count}")
            elif batch_idx == 5:
                print("...")
            elif batch_idx >= 95:  # Show last 5 batches
                print(f"Batch {batch_idx}: Real={real_count}, Fake={fake_count}")
            batch_count += 1
        
        if batch_count > 0:
            avg_real = np.mean(real_counts)
            avg_fake = np.mean(fake_counts)
            std_real = np.std(real_counts)
            std_fake = np.std(fake_counts)
            print(f"\nTested {batch_count} batches:")
            print(f"Average per batch: Real={avg_real:.1f}±{std_real:.1f}, Fake={avg_fake:.1f}±{std_fake:.1f}")
            print(f"Perfect balance: {'✓' if abs(avg_real - avg_fake) < 0.1 else '✗'}")
            print(f"Consistent balance: {'✓' if std_real < 1 and std_fake < 1 else '✗'}")
            print(f"Total samples tested: {batch_count * 32}")
            
    except Exception as e:
        print(f"✗ Error creating dataloader: {e}")

def show_usage():
    """Show how to run the project"""
    print("\n" + "="*50)
    print("HOW TO RUN THE PROJECT")
    print("="*50)
    
    print("\n1. TRAIN DETECTION MODEL:")
    print("   python main.py --mode train_detect")
    
    print("\n2. TRAIN SEGMENTATION MODEL:")
    print("   python main.py --mode train_seg")
    
    print("\n3. EVALUATE ON IMAGE:")
    print("   python main.py --mode eval --image path/to/image.png")
    
    print("\n4. EXAMPLE WITH SAMPLE IMAGE:")
    print("   python main.py --mode eval --image data/wacv_data/images/60000.png")
    
    print("\n" + "="*50)
    print("PROJECT STRUCTURE")
    print("="*50)
    print("deepfake_project/")
    print("├── src/")
    print("│   ├── models/     # Detection & segmentation models")
    print("│   ├── data/       # Data loading & balancing")
    print("│   ├── training/   # Training classes")
    print("│   └── explainability/ # Grad-CAM & LIME")
    print("├── scripts/        # CLI entry points")
    print("├── configs/        # Configuration files")
    print("├── data/           # Dataset")
    print("├── outputs/        # Results & checkpoints")
    print("└── requirements.txt")
    
    print("\n" + "="*50)
    print("BALANCING FEATURES")
    print("="*50)
    print("✓ BalancedSampler: Ensures 50/50 real/fake per batch")
    print("✓ Automatic dataset split filtering")
    print("✓ Configurable via balanced_batch flag")
    print("✓ Mixed precision training for speed")
    print("✓ cuDNN benchmarking enabled")

if __name__ == "__main__":
    test_dataset_balance()
    show_usage() 