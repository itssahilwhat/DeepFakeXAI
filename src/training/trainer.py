import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
import sys
import os
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.config import lr, epochs, dir_ckpt, seg_loss_w
from src.data.clean_dataset import make_clean_loader
import os

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def to_device(model):
    """Move model to appropriate device (GPU/CPU)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return model.to(device), device

def dice_loss(pred, target, smooth=1e-6):
    """Compute Dice loss for segmentation"""
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice

# ============================================================================
# SEGMENTATION TRAINER
# ============================================================================

class SegmentationTrainer:
    """Trainer for deepfake segmentation models"""
    
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device, checkpoint_dir, patience=5, resume_from=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.patience = patience
        
        # Training state
        self.best_iou = 0.0
        self.best_dice = 0.0
        self.best_pixel_acc = 0.0
        self.patience_counter = 0
        self.start_epoch = 0
        
        # Mixed precision training
        self.scaler = GradScaler('cuda') if torch.cuda.is_available() else None
        
        # Loss functions
        self.focal_criterion = self.focal_bce_loss
        self.dice_criterion = self.dice_loss
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)
        else:
            # Try to load from latest checkpoint
            self.load_latest_checkpoint()
        
    def dice_loss(self, pred, target, smooth=1e-6):
        """Compute Dice loss for segmentation"""
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return 1 - dice
    
    def focal_bce_loss(self, pred, target, alpha=0.25, gamma=2.0):
        """Focal BCE loss to focus on hard examples"""
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        return focal_loss.mean()
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Handle different batch formats
            if len(batch) == 3:
                images, masks, labels = batch
            else:
                images, labels, paths = batch
                # Skip if no masks available
                continue
                
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.scaler:
                with autocast('cuda'):
                    outputs = self.model(images)
                    
                    # Handle multi-scale outputs (UNet training mode)
                    if isinstance(outputs, (tuple, list)):
                        # Clamp outputs to prevent extreme values
                        outputs = tuple(torch.clamp(out, -20, 20) for out in outputs)
                        out_main, out_2x, out_4x = outputs
                        
                        # Resize GT masks to match each scale
                        mask_main = F.interpolate(masks, size=out_main.shape[-2:], mode='nearest')
                        mask_2x = F.interpolate(masks, size=out_2x.shape[-2:], mode='nearest')
                        mask_4x = F.interpolate(masks, size=out_4x.shape[-2:], mode='nearest')
                        
                        # Compute weighted multi-scale loss
                        loss_main = self.focal_criterion(out_main, mask_main) + self.dice_criterion(out_main, mask_main)
                        loss_2x = self.focal_criterion(out_2x, mask_2x) + self.dice_criterion(out_2x, mask_2x)
                        loss_4x = self.focal_criterion(out_4x, mask_4x) + self.dice_criterion(out_4x, mask_4x)
                        loss = 1.0 * loss_main + 0.5 * loss_2x + 0.25 * loss_4x
                    else:
                        # Single output
                        outputs = torch.clamp(outputs, -20, 20)
                        loss = self.focal_criterion(outputs, masks) + self.dice_criterion(outputs, masks)
                
                # Backward pass with mixed precision
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Regular training without mixed precision
                outputs = self.model(images)
                
                if isinstance(outputs, (tuple, list)):
                    outputs = tuple(torch.clamp(out, -20, 20) for out in outputs)
                    out_main, out_2x, out_4x = outputs
                    
                    mask_main = F.interpolate(masks, size=out_main.shape[-2:], mode='nearest')
                    mask_2x = F.interpolate(masks, size=out_2x.shape[-2:], mode='nearest')
                    mask_4x = F.interpolate(masks, size=out_4x.shape[-2:], mode='nearest')
                    
                    loss_main = self.focal_criterion(out_main, mask_main) + self.dice_criterion(out_main, mask_main)
                    loss_2x = self.focal_criterion(out_2x, mask_2x) + self.dice_criterion(out_2x, mask_2x)
                    loss_4x = self.focal_criterion(out_4x, mask_4x) + self.dice_criterion(out_4x, mask_4x)
                    loss = 1.0 * loss_main + 0.5 * loss_2x + 0.25 * loss_4x
                else:
                    outputs = torch.clamp(outputs, -20, 20)
                    loss = self.focal_criterion(outputs, masks) + self.dice_criterion(outputs, masks)
                
                # Regular backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_iou = 0.0
        total_dice = 0.0
        total_pixel_acc = 0.0
        num_samples = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for batch_idx, batch in enumerate(pbar):
                # Handle different batch formats
                if len(batch) == 3:
                    images, masks, labels = batch
                else:
                    images, labels, paths = batch
                    continue
                
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                
                # Handle multi-scale outputs
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]  # Use main output for validation
                
                # Convert to binary predictions
                pred_masks = torch.sigmoid(outputs) > 0.5
                
                # Compute metrics for each sample
                for i in range(images.size(0)):
                    pred = pred_masks[i]
                    target = masks[i] > 0.5
                    
                    # IoU (Intersection over Union)
                    intersection = (pred & target).sum()
                    union = (pred | target).sum()
                    iou = intersection / union if union > 0 else 0.0
                    
                    # Dice coefficient
                    dice = (2 * intersection) / (pred.sum() + target.sum()) if (pred.sum() + target.sum()) > 0 else 0.0
                    
                    # Pixel accuracy
                    pixel_acc = (pred == target).float().mean()
                    
                    total_iou += iou
                    total_dice += dice
                    total_pixel_acc += pixel_acc
                    num_samples += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'IoU': f'{total_iou/num_samples:.4f}',
                    'Dice': f'{total_dice/num_samples:.4f}'
                })
        
        # Return average metrics
        return {
            'iou': total_iou / num_samples if num_samples > 0 else 0.0,
            'dice': total_dice / num_samples if num_samples > 0 else 0.0,
            'pixel_acc': total_pixel_acc / num_samples if num_samples > 0 else 0.0,
            'num_samples': num_samples
        }
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_iou': self.best_iou,
            'best_dice': self.best_dice,
            'best_pixel_acc': self.best_pixel_acc,
            'metrics': metrics
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'seg_latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint if improved
        if is_best:
            best_path = self.checkpoint_dir / 'seg_best.pth'
            torch.save(checkpoint, best_path)
            print(f"🏆 New best model saved! IoU: {metrics['iou']:.4f}")
        
        # Save epoch checkpoint
        epoch_path = self.checkpoint_dir / f'seg_epoch_{epoch+1}.pth'
        torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint from specific path"""
        if not Path(checkpoint_path).exists():
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            return False
            
        print(f"📂 Loading checkpoint from: {checkpoint_path}")
        try:
            # Try loading with weights_only=False for backward compatibility
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}")
            print(f"   Starting fresh training instead...")
            return False
        
        # Load model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_iou = checkpoint.get('best_iou', 0.0)
        self.best_dice = checkpoint.get('best_dice', 0.0)
        self.best_pixel_acc = checkpoint.get('best_pixel_acc', 0.0)
        
        print(f"✅ Resumed from epoch {checkpoint['epoch']+1}")
        print(f"   Best IoU: {self.best_iou:.4f}")
        print(f"   Best Dice: {self.best_dice:.4f}")
        return True
    
    def load_latest_checkpoint(self):
        """Load from latest checkpoint if available"""
        latest_path = self.checkpoint_dir / 'seg_latest.pth'
        if latest_path.exists():
            print(f"📂 Found latest checkpoint, resuming training...")
            return self.load_checkpoint(latest_path)
        else:
            print(f"📂 No previous checkpoint found, starting fresh training")
            return False
    
    def fit(self, num_epochs, stage='stage2'):
        """Main training loop"""
        print(f"\n🚀 Starting segmentation training for {num_epochs} epochs")
        print(f"   Stage: {stage}")
        print(f"   Device: {self.device}")
        print(f"   Early stopping patience: {self.patience}")
        print(f"   Starting from epoch: {self.start_epoch + 1}")
        
        for epoch in range(self.start_epoch, num_epochs):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1}/{num_epochs}")
            print(f"{'='*60}")
            
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate()
            
            # Learning rate step
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Print results
            print(f"\n📊 EPOCH {epoch+1} RESULTS:")
            print(f"{'─'*40}")
            print(f"🎯 Training Loss:     {train_loss:.4f}")
            print(f"🔍 Validation Metrics:")
            print(f"  IoU:               {val_metrics['iou']:.4f}")
            print(f"  Dice:              {val_metrics['dice']:.4f}")
            print(f"  Pixel Accuracy:    {val_metrics['pixel_acc']:.4f}")
            print(f"  Samples processed: {val_metrics['num_samples']}")
            print(f"📚 Learning Rate:     {current_lr:.6f}")
            
            # Check for improvement
            current_iou = val_metrics['iou']
            is_best = current_iou > self.best_iou
            
            if is_best:
                self.best_iou = current_iou
                self.best_dice = val_metrics['dice']
                self.best_pixel_acc = val_metrics['pixel_acc']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\n⏰ Early stopping triggered after {self.patience} epochs without improvement")
                break
            
            print(f"🏆 Best IoU so far: {self.best_iou:.4f} (patience: {self.patience_counter}/{self.patience})")
        
        # Return best metrics
        return {
            'iou': self.best_iou,
            'dice': self.best_dice,
            'pixel_acc': self.best_pixel_acc
        }