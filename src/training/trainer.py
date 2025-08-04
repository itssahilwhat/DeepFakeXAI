import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import sys
import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.config import lr, epochs, dir_ckpt, seg_loss_w
from src.models.models import DetectionModel, SegmentationModel
from src.data.data_utils import get_dataloader
import os

def to_device(model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return model.to(device), device

class DetectionTrainer:
    def __init__(self):
        self.model, self.device = to_device(DetectionModel())
        self.opt = AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        self.sched = CosineAnnealingLR(self.opt, T_max=epochs)
        self.scaler = GradScaler()
        self.crit = nn.CrossEntropyLoss()
        self.best_acc = 0
        self.best_f1 = 0
    def train_epoch(self, loader):
        self.model.train()
        for x, y, _ in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.opt.zero_grad()
            with autocast():
                out = self.model(x)
                loss = self.crit(out, y)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()
        self.sched.step()
    def validate(self, loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for x, y, _ in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                probs = torch.softmax(out, dim=1)
                preds = out.argmax(1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Fake class probability
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        accuracy = (all_preds == all_labels).mean()
        
        # Precision, Recall, F1 for Fake class (class 1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        
        # ROC-AUC
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.5  # Fallback if only one class present
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    def fit(self):
        train_loader = get_dataloader('train')
        val_loader = get_dataloader('val')
        
        for epoch in range(epochs):
            self.train_epoch(train_loader)
            metrics = self.validate(val_loader)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1']:.4f}")
            print(f"  ROC-AUC: {metrics['auc']:.4f}")
            
            # Save best model based on F1-score (better than accuracy for imbalanced data)
            if metrics['f1'] > self.best_f1:
                self.best_f1 = metrics['f1']
                torch.save(self.model.state_dict(), os.path.join(dir_ckpt, 'detect.pth'))
                print(f"  ✓ New best model saved (F1: {metrics['f1']:.4f})")
            print()

class SegmentationTrainer:
    def __init__(self):
        self.model, self.device = to_device(SegmentationModel())
        self.opt = AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        self.sched = CosineAnnealingLR(self.opt, T_max=epochs)
        self.scaler = GradScaler()
        self.crit = nn.BCEWithLogitsLoss(reduction='none')
        self.best_iou = 0
    def train_epoch(self, loader):
        self.model.train()
        for x, _, mask in loader:
            x = x.to(self.device)
            mask = mask.to(self.device)
            self.opt.zero_grad()
            with autocast():
                out = self.model(x)
                # Only compute loss on non-empty masks
                loss_per_pixel = self.crit(out, mask)
                # Average over valid pixels (where mask > 0)
                valid_pixels = (mask > 0).float()
                if valid_pixels.sum() > 0:
                    loss = (loss_per_pixel * valid_pixels).sum() / valid_pixels.sum()
                else:
                    loss = loss_per_pixel.mean()  # Fallback for empty masks
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()
        self.sched.step()
    def validate(self, loader):
        self.model.eval()
        iou_scores = []
        dice_scores = []
        pixel_accuracies = []
        
        with torch.no_grad():
            for x, _, mask in loader:
                x = x.to(self.device)
                mask = mask.to(self.device)
                out = self.model(x)
                pred = (torch.sigmoid(out) > 0.5).float()
                
                # Only compute metrics on non-empty masks
                if mask.sum() > 0:
                    # IoU
                    inter = (pred * mask).sum().item()
                    union = (pred + mask).clamp(0,1).sum().item()
                    iou = inter / (union + 1e-8)
                    iou_scores.append(iou)
                    
                    # Dice
                    dice = 2*inter / (pred.sum().item() + mask.sum().item() + 1e-8)
                    dice_scores.append(dice)
                    
                    # Pixel Accuracy
                    correct_pixels = ((pred == mask) & (mask > 0)).sum().item()
                    total_pixels = (mask > 0).sum().item()
                    pixel_acc = correct_pixels / (total_pixels + 1e-8)
                    pixel_accuracies.append(pixel_acc)
        
        # Calculate averages
        avg_iou = np.mean(iou_scores) if iou_scores else 0
        avg_dice = np.mean(dice_scores) if dice_scores else 0
        avg_pixel_acc = np.mean(pixel_accuracies) if pixel_accuracies else 0
        
        return {
            'iou': avg_iou,
            'dice': avg_dice,
            'pixel_accuracy': avg_pixel_acc,
            'num_samples': len(iou_scores)
        }
    def fit(self):
        train_loader = get_dataloader('train')
        val_loader = get_dataloader('val')
        
        for epoch in range(epochs):
            self.train_epoch(train_loader)
            metrics = self.validate(val_loader)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  IoU: {metrics['iou']:.4f}")
            print(f"  Dice: {metrics['dice']:.4f}")
            print(f"  Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
            print(f"  Samples with masks: {metrics['num_samples']}")
            
            if metrics['iou'] > self.best_iou:
                self.best_iou = metrics['iou']
                torch.save(self.model.state_dict(), os.path.join(dir_ckpt, 'seg.pth'))
                print(f"  ✓ New best model saved (IoU: {metrics['iou']:.4f})")
            print()