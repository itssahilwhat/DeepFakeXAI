#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb
import json
from pathlib import Path
from abc import ABC, abstractmethod
from datetime import datetime
import time

from src.core.config import Config
from src.preprocessing.dataset import get_dataloader
from src.core.losses import FocalLoss, LabelSmoothingCrossEntropy
from src.evaluation.metrics import ClassificationMetrics, LocalizationMetrics

class BaseTrainer(ABC):
    """Base trainer class for all deepfake detection models with comprehensive metrics"""
    
    def __init__(self, config: Config, model_name: str, resume_from: str = None):
        self.config = config
        self.device = config.DEVICE
        self.model_name = model_name
        self.resume_from = resume_from
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Initialize dataloaders with balanced sampling
        self.train_loader = get_dataloader('train', config, supervision_filter=None)
        self.val_loader = get_dataloader('valid', config, supervision_filter=None)
        self.test_loader = get_dataloader('test', config, supervision_filter=None)
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=config.MIN_LR
        )
        
        # Initialize loss function
        self.criterion = self._create_loss_function()
        
        # Initialize metrics calculators
        self.cls_metrics = ClassificationMetrics()
        self.loc_metrics = LocalizationMetrics()
        
        # Training state
        self.start_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.best_val_auc = 0.0
        self.epoch = 0
        
        # Comprehensive metrics storage
        self.metrics_history = {
            'train': [],
            'val': [],
            'test': []
        }
        
        # Setup logging
        self._setup_logging()
        
        # Resume from checkpoint if specified
        if resume_from:
            self._resume_from_checkpoint(resume_from)
    
    @abstractmethod
    def _create_model(self):
        """Create the model instance - to be implemented by subclasses"""
        pass
    
    def _create_loss_function(self):
        """Create loss function with class balancing"""
        if self.config.USE_FOCAL_LOSS:
            # Calculate pos_weight for class balancing
            train_labels = self.train_loader.dataset.labels
            real_count = sum(1 for label in train_labels if label == 0)
            fake_count = sum(1 for label in train_labels if label == 1)
            pos_weight = torch.tensor(real_count / fake_count) if fake_count > 0 else torch.tensor(1.0)
            
            print(f"Class distribution - Real: {real_count}, Fake: {fake_count}")
            print(f"Using pos_weight: {pos_weight.item():.3f}")
            
            return FocalLoss(
                alpha=self.config.FOCAL_ALPHA, 
                gamma=self.config.FOCAL_GAMMA,
                pos_weight=pos_weight
            )
        else:
            return LabelSmoothingCrossEntropy(smoothing=self.config.LABEL_SMOOTHING)
    
    def _setup_logging(self):
        """Setup wandb logging"""
        wandb.init(
            project="deepfake-detection",
            name=f"{self.model_name}-classifier",
            config=vars(self.config),
            resume=True
        )
    
    def _resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from a specific checkpoint"""
        print(f"Resuming from checkpoint: {checkpoint_path}")
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer and scheduler state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training state
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
            self.best_val_f1 = checkpoint.get('best_val_f1', 0.0)
            self.best_val_auc = checkpoint.get('best_val_auc', 0.0)
            
            # Load metrics history
            self.metrics_history = checkpoint.get('metrics_history', {
                'train': [], 'val': [], 'test': []
            })
            
            print(f"Resumed from epoch {checkpoint['epoch']}")
            print(f"Best validation accuracy: {self.best_val_acc:.4f}")
            print(f"Best validation F1: {self.best_val_f1:.4f}")
            print(f"Best validation AUC: {self.best_val_auc:.4f}")
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
    
    def _calculate_comprehensive_metrics(self, predictions, labels, probabilities, masks=None, pred_masks=None):
        """Calculate comprehensive metrics for all categories"""
        metrics = {}
        
        # 1. Deepfake Detection Metrics
        cls_metrics = self.cls_metrics.calculate_metrics(predictions, labels, probabilities)
        metrics.update({
            'accuracy': cls_metrics['accuracy'],
            'precision': cls_metrics['precision'],
            'recall': cls_metrics['recall'],
            'f1': cls_metrics['f1'],
            'auc': cls_metrics['auc'],
            'ap': cls_metrics['ap'],
            'confusion_matrix': cls_metrics['confusion_matrix'],
            'eer': self._calculate_eer(probabilities, labels),
            'ece': self._calculate_ece(probabilities, labels)
        })
        
        # 2. Localization Metrics (if masks available)
        if masks is not None and pred_masks is not None:
            loc_metrics = self.loc_metrics.calculate_metrics(pred_masks, masks)
            metrics.update({
                'iou_mean': loc_metrics['iou_mean'],
                'iou_std': loc_metrics['iou_std'],
                'dice_mean': loc_metrics['dice_mean'],
                'dice_std': loc_metrics['dice_std'],
                'pbca_mean': loc_metrics['pbca_mean'],
                'pbca_std': loc_metrics['pbca_std'],
                'mae': self._calculate_mae(pred_masks, masks),
                'boundary_iou': self._calculate_boundary_iou(pred_masks, masks)
            })
        
        # 3. XAI Metrics (if model supports it)
        if hasattr(self.model, 'get_gradcam'):
            xai_metrics = self._calculate_xai_metrics()
            metrics.update(xai_metrics)
        
        # 4. Performance Metrics
        performance_metrics = self._calculate_performance_metrics()
        metrics.update(performance_metrics)
        
        return metrics
    
    def _calculate_eer(self, probabilities, labels):
        """Calculate Equal Error Rate"""
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(labels, probabilities)
        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
        eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
        return eer
    
    def _calculate_ece(self, probabilities, labels, n_bins=15):
        """Calculate Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = np.logical_and(probabilities > bin_lower, probabilities <= bin_upper)
            bin_size = np.sum(in_bin)
            
            if bin_size > 0:
                bin_accuracy = np.sum(labels[in_bin]) / bin_size
                bin_confidence = np.mean(probabilities[in_bin])
                ece += bin_size * np.abs(bin_accuracy - bin_confidence)
        
        return ece / len(probabilities)
    
    def _calculate_mae(self, pred_masks, gt_masks):
        """Calculate Mean Absolute Error for masks"""
        return np.mean(np.abs(pred_masks - gt_masks))
    
    def _calculate_boundary_iou(self, pred_masks, gt_masks):
        """Calculate Boundary IoU"""
        # Simplified boundary calculation
        from scipy import ndimage
        pred_boundaries = ndimage.binary_erosion(pred_masks) != pred_masks
        gt_boundaries = ndimage.binary_erosion(gt_masks) != gt_masks
        
        intersection = np.logical_and(pred_boundaries, gt_boundaries).sum()
        union = np.logical_or(pred_boundaries, gt_boundaries).sum()
        return intersection / (union + 1e-8)
    
    def _calculate_xai_metrics(self):
        """Calculate XAI metrics if model supports GradCAM"""
        # Placeholder for XAI metrics
        return {
            'faithfulness': 0.0,
            'sensitivity_n': 0.0,
            'localization_error': 0.0
        }
    
    def _calculate_performance_metrics(self):
        """Calculate performance metrics"""
        # Model size
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        model_size_mb = (param_size + buffer_size) / (1024 * 1024)
        
        return {
            'model_size_mb': model_size_mb,
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
    
    def train_epoch(self):
        """Train for one epoch with comprehensive metrics"""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probs = []
        all_masks = []
        all_pred_masks = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, (images, masks, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.long().to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            loss, logits, pred_masks = self._forward_pass(images, masks, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
            
            self.optimizer.step()
            
            # Collect metrics
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits.data, 1)
            
            all_predictions.extend(predicted.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(probs[:, 1].detach().cpu().numpy())
            all_masks.extend(masks.detach().cpu().numpy())
            if pred_masks is not None:
                all_pred_masks.extend(pred_masks.detach().cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * np.mean(np.array(all_predictions) == np.array(all_labels)):.2f}%'
            })
            
            # Log to wandb
            if batch_idx % 50 == 0:
                wandb.log({
                    'train_loss': loss.item(),
                    'train_acc': 100 * np.mean(np.array(all_predictions) == np.array(all_labels)),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(
            all_predictions, all_labels, all_probs, 
            all_masks if all_masks else None,
            all_pred_masks if all_pred_masks else None
        )
        
        return total_loss / len(self.train_loader), metrics
    
    def _forward_pass(self, images, masks, labels):
        """Forward pass - to be overridden by subclasses if needed"""
        logits = self.model(images)
        loss = self.criterion(logits, labels)
        return loss, logits, None
    
    def validate(self, loader, split='val'):
        """Validate model with comprehensive metrics"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probs = []
        all_masks = []
        all_pred_masks = []
        
        with torch.no_grad():
            for images, masks, labels in tqdm(loader, desc=f"Validating {split}"):
                images = images.to(self.device)
                labels = labels.long().to(self.device)
                masks = masks.to(self.device)
                
                loss, logits, pred_masks = self._validation_forward(images, masks, labels)
                
                total_loss += loss.item()
                probs = torch.softmax(logits, dim=1)
                _, predicted = torch.max(logits.data, 1)
                
                all_predictions.extend(predicted.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())
                all_probs.extend(probs[:, 1].detach().cpu().numpy())
                all_masks.extend(masks.detach().cpu().numpy())
                if pred_masks is not None:
                    all_pred_masks.extend(pred_masks.detach().cpu().numpy())
        
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(
            all_predictions, all_labels, all_probs,
            all_masks if all_masks else None,
            all_pred_masks if all_pred_masks else None
        )
        
        avg_loss = total_loss / len(loader)
        metrics['loss'] = avg_loss
        
        return avg_loss, metrics
    
    def _validation_forward(self, images, masks, labels):
        """Validation forward pass - to be overridden by subclasses if needed"""
        logits = self.model(images)
        loss = self.criterion(logits, labels)
        return loss, logits, None
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save comprehensive checkpoint with all metrics"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_f1': self.best_val_f1,
            'best_val_auc': self.best_val_auc,
            'metrics_history': self.metrics_history,
            'config': vars(self.config),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save epoch checkpoint
        epoch_path = os.path.join(self.config.CHECKPOINT_DIR, f'{self.model_name}_epoch_{epoch}.pth')
        torch.save(checkpoint, epoch_path)
        
        # Save latest checkpoint
        latest_path = os.path.join(self.config.CHECKPOINT_DIR, f'{self.model_name}_latest.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config.CHECKPOINT_DIR, f'{self.model_name}_best.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved! Validation accuracy: {self.best_val_acc:.4f}")
        
        # Save metrics to JSON for easy analysis
        metrics_path = os.path.join(self.config.CHECKPOINT_DIR, f'{self.model_name}_epoch_{epoch}_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Epoch {epoch} checkpoint saved: {epoch_path}")
    
    def _print_epoch_metrics(self, epoch, train_metrics, val_metrics):
        """Print comprehensive epoch metrics"""
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch} RESULTS")
        print(f"{'='*80}")
        
        # Training metrics
        print(f"\n📊 TRAINING METRICS:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"  Precision: {train_metrics['precision']:.4f}")
        print(f"  Recall: {train_metrics['recall']:.4f}")
        print(f"  F1: {train_metrics['f1']:.4f}")
        print(f"  AUC: {train_metrics['auc']:.4f}")
        print(f"  AP: {train_metrics['ap']:.4f}")
        print(f"  EER: {train_metrics['eer']:.4f}")
        print(f"  ECE: {train_metrics['ece']:.4f}")
        
        if 'iou_mean' in train_metrics:
            print(f"  IoU: {train_metrics['iou_mean']:.4f} ± {train_metrics['iou_std']:.4f}")
            print(f"  Dice: {train_metrics['dice_mean']:.4f} ± {train_metrics['dice_std']:.4f}")
            print(f"  PBCA: {train_metrics['pbca_mean']:.4f} ± {train_metrics['pbca_std']:.4f}")
        
        # Validation metrics
        print(f"\n🎯 VALIDATION METRICS:")
        print(f"  Loss: {val_metrics['loss']:.4f}")
        print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall: {val_metrics['recall']:.4f}")
        print(f"  F1: {val_metrics['f1']:.4f}")
        print(f"  AUC: {val_metrics['auc']:.4f}")
        print(f"  AP: {val_metrics['ap']:.4f}")
        print(f"  EER: {val_metrics['eer']:.4f}")
        print(f"  ECE: {val_metrics['ece']:.4f}")
        
        if 'iou_mean' in val_metrics:
            print(f"  IoU: {val_metrics['iou_mean']:.4f} ± {val_metrics['iou_std']:.4f}")
            print(f"  Dice: {val_metrics['dice_mean']:.4f} ± {val_metrics['dice_std']:.4f}")
            print(f"  PBCA: {val_metrics['pbca_mean']:.4f} ± {val_metrics['pbca_std']:.4f}")
        
        # Performance metrics
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"  Model Size: {val_metrics['model_size_mb']:.2f} MB")
        print(f"  Parameters: {val_metrics['num_parameters']:,}")
        print(f"  Trainable: {val_metrics['trainable_parameters']:,}")
    
    def train(self):
        """Main training loop with comprehensive metrics and checkpointing"""
        print(f"Starting {self.model_name} training...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Starting from epoch: {self.start_epoch}")
        
        for epoch in range(self.start_epoch, self.config.EPOCHS):
            self.epoch = epoch
            
            # Train
            train_loss, train_metrics = self.train_epoch()
            
            # Validate
            val_loss, val_metrics = self.validate(self.val_loader, 'val')
            
            # Update scheduler
            self.scheduler.step()
            
            # Update best metrics
            is_best = False
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                is_best = True
            
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
            
            if val_metrics['auc'] > self.best_val_auc:
                self.best_val_auc = val_metrics['auc']
            
            # Store metrics history
            self.metrics_history['train'].append(train_metrics)
            self.metrics_history['val'].append(val_metrics)
            
            # Save checkpoint
            combined_metrics = {
                'train': train_metrics,
                'val': val_metrics,
                'epoch': epoch
            }
            self.save_checkpoint(epoch, combined_metrics, is_best)
            
            # Print comprehensive metrics
            self._print_epoch_metrics(epoch, train_metrics, val_metrics)
            
            # Log to wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'train_acc': train_metrics['accuracy'],
                'val_acc': val_metrics['accuracy'],
                'train_f1': train_metrics['f1'],
                'val_f1': val_metrics['f1'],
                'train_auc': train_metrics['auc'],
                'val_auc': val_metrics['auc'],
                'best_val_acc': self.best_val_acc,
                'best_val_f1': self.best_val_f1,
                'best_val_auc': self.best_val_auc
            })
            
            # Early stopping check
            if epoch > 20 and val_metrics['accuracy'] < 0.5:
                print("Early stopping due to poor performance")
                break
        
        # Final evaluation on test set
        print("\nFinal evaluation on test set...")
        test_loss, test_metrics = self.validate(self.test_loader, 'test')
        self.metrics_history['test'].append(test_metrics)
        
        print(f"\n🏁 FINAL TEST RESULTS:")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  F1: {test_metrics['f1']:.4f}")
        print(f"  AUC: {test_metrics['auc']:.4f}")
        
        # Save final metrics
        final_metrics = {
            'train': self.metrics_history['train'][-1],
            'val': self.metrics_history['val'][-1],
            'test': test_metrics,
            'best_val_acc': self.best_val_acc,
            'best_val_f1': self.best_val_f1,
            'best_val_auc': self.best_val_auc
        }
        
        final_path = os.path.join(self.config.CHECKPOINT_DIR, f'{self.model_name}_final_metrics.json')
        with open(final_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        wandb.finish() 