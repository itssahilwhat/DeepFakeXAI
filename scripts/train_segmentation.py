#!/usr/bin/env python3
"""
Segmentation training script using the SegmentationTrainer class.
Trains UNet models for deepfake segmentation with mask supervision.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
import argparse
import random
import os
import numpy as np
from tqdm import tqdm

from src.models.models import UNet
from src.data.clean_dataset import make_clean_loader
from src.training.trainer import SegmentationTrainer
from configs.config import data_root

def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def main():
    parser = argparse.ArgumentParser(description='Train segmentation model for deepfake detection')
    parser.add_argument('--train-manifest', type=str, required=True,
                       help='Path to training manifest CSV')
    parser.add_argument('--val-manifest', type=str, required=True,
                       help='Path to validation manifest CSV')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=25,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set reproducibility
    set_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"📊 Loading training data from {args.train_manifest}...")
    train_loader = make_clean_loader(
        args.train_manifest, data_root, 
        batch_size=args.batch_size, 
        is_train=True, 
        balance=True
    )
    
    print(f"📊 Loading validation data from {args.val_manifest}...")
    val_loader = make_clean_loader(
        args.val_manifest, data_root, 
        batch_size=args.batch_size, 
        is_train=False, 
        balance=False
    )
    
    print(f"✅ Training batches: {len(train_loader)}")
    print(f"✅ Validation batches: {len(val_loader)}")
    
    # Initialize model
    print(f"🏗️  Initializing UNet model...")
    model = UNet(base_ch=64, use_se=True)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Initialize trainer
    print(f"🎯 Initializing SegmentationTrainer...")
    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=checkpoint_dir,
        patience=args.patience,
        resume_from=args.resume_from
    )
    
    # Start training
    print(f"\n🚀 Starting segmentation training...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Weight decay: {args.weight_decay}")
    print(f"   Patience: {args.patience}")
    print(f"   Checkpoint dir: {checkpoint_dir}")
    
    try:
        best_metrics = trainer.fit(num_epochs=args.epochs)
        
        print(f"\n🎉 Training completed successfully!")
        print(f"🏆 Best metrics:")
        print(f"   IoU: {best_metrics['iou']:.4f}")
        print(f"   Dice: {best_metrics['dice']:.4f}")
        print(f"   Pixel Accuracy: {best_metrics['pixel_acc']:.4f}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise
    
    print(f"\n📁 Checkpoints saved in: {checkpoint_dir}")

if __name__ == '__main__':
    main()
