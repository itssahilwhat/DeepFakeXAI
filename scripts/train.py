#!/usr/bin/env python3

"""
Unified training script for deepfake detection models.
Supports multiple backbones, U-Net segmentation, and hyperparameter tuning.
"""

import os
import sys
import argparse
import torch
import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append('.')

from src.core.config import Config
from src.training.trainers import GradCAMTrainer, PatchForensicsTrainer, MobileNetV3Trainer, UNetTrainer, EfficientNetTrainer, ResNetSETrainer
from src.detection.deepfake_models import DeepfakeDetectionModel

def main():
    parser = argparse.ArgumentParser(description='Train deepfake detection models')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['gradcam', 'patch', 'mobilenetv3', 'unet', 'efficientnet', 'resnet34se', 'attention', 'multitask'],
                       help='Model type to train')
    parser.add_argument('--backbone', type=str, default='xception',
                       choices=['xception', 'efficientnet_b0', 'resnet34', 'resnet50'],
                       help='Backbone architecture')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--resume-latest', action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--sweep', action='store_true', help='Run W&B hyperparameter sweep')
    parser.add_argument('--sweep-count', type=int, default=10, help='Number of sweep runs')
    
    parser.add_argument('--detection-weight', type=float, default=1.0, help='Detection loss weight')
    parser.add_argument('--segmentation-weight', type=float, default=1.0, help='Segmentation loss weight')
    parser.add_argument('--attention-weight', type=float, default=0.1, help='Attention loss weight')
    
    parser.add_argument('--aug-strength', type=str, default='medium',
                       choices=['weak', 'medium', 'strong'], help='Augmentation strength')
    
    args = parser.parse_args()
    
    config = Config()
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    config.WEIGHT_DECAY = args.weight_decay
    config.AUG_STRENGTH = args.aug_strength
    
    if args.sweep:
        sweep_config = {
            'method': 'bayes',
            'metric': {'name': 'val_f1', 'goal': 'maximize'},
            'parameters': {
                'learning_rate': {'min': 1e-5, 'max': 1e-3, 'distribution': 'log_uniform'},
                'batch_size': {'values': [16, 32, 64]},
                'weight_decay': {'min': 1e-5, 'max': 1e-3, 'distribution': 'log_uniform'},
                'detection_weight': {'min': 0.5, 'max': 2.0},
                'segmentation_weight': {'min': 0.5, 'max': 2.0},
                'attention_weight': {'min': 0.05, 'max': 0.5}
            }
        }
        
        sweep_id = wandb.sweep(sweep_config, project="deepfake-detection")
        wandb.agent(sweep_id, train_model, count=args.sweep_count)
    else:
        train_model(config, args)

def train_model(config=None, args=None):
    if config is None:
        config = Config()
    
    wandb.init(project="deepfake-detection", config=vars(args) if args else {})
    
    if args.model == 'gradcam':
        trainer = GradCAMTrainer(config)
    elif args.model == 'patch':
        trainer = PatchForensicsTrainer(config)
    elif args.model == 'mobilenetv3':
        trainer = MobileNetV3Trainer(config)
    elif args.model == 'unet':
        trainer = UNetTrainer(config)
    elif args.model == 'efficientnet':
        trainer = EfficientNetTrainer(config, backbone=args.backbone)
    elif args.model == 'resnet34se':
        trainer = ResNetSETrainer(config, backbone='resnet34')
    elif args.model == 'attention':
        trainer = AttentionTrainer(config)
    elif args.model == 'multitask':
        trainer = MultiTaskTrainer(config, backbone=args.backbone)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    if args.resume:
        trainer.resume_from_checkpoint(args.resume)
    elif args.resume_latest:
        trainer.resume_from_latest()
    
    trainer.train()

if __name__ == "__main__":
    main() 