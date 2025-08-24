#!/usr/bin/env python3
"""
Clean training script using deduplicated manifests for realistic deepfake detection.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import random
import os
from datetime import datetime
import sklearn.metrics

from src.models.models import VanillaCNN
from src.data.clean_dataset import make_clean_loader, create_combined_clean_dataset

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_metrics(outputs, labels):
    """Compute comprehensive classification metrics."""
    probs = torch.softmax(outputs, dim=1)
    predictions = torch.argmax(outputs, dim=1)
    
    # Basic accuracy
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    
    # Per-class accuracy
    real_mask = (labels == 0)
    fake_mask = (labels == 1)
    
    real_acc = 0.0
    fake_acc = 0.0
    if real_mask.sum() > 0:
        real_acc = (predictions[real_mask] == labels[real_mask]).float().mean().item()
    if fake_mask.sum() > 0:
        fake_acc = (predictions[fake_mask] == labels[fake_mask]).float().mean().item()
    
    # Confusion matrix components
    tp = ((predictions == 1) & (labels == 1)).sum().item()  # True positives
    tn = ((predictions == 0) & (labels == 0)).sum().item()  # True negatives
    fp = ((predictions == 1) & (labels == 0)).sum().item()  # False positives
    fn = ((predictions == 0) & (labels == 1)).sum().item()  # False negatives
    
    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # ROC-AUC
    try:
        if len(np.unique(labels.cpu().numpy())) > 1:  # Only compute if both classes present
            auc = sklearn.metrics.roc_auc_score(labels.cpu().numpy(), probs[:,1].detach().cpu().numpy())
        else:
            auc = 0.5  # Random performance when only one class
    except Exception:
        auc = 0.5
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = recall  # Same as recall
    
    return {
        'accuracy': accuracy,
        'real_acc': real_acc,
        'fake_acc': fake_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }

def print_confusion_matrix(metrics):
    """Print formatted confusion matrix."""
    print(f"📊 Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Real  Fake")
    print(f"Actual Real   {int(metrics['tn']):4d}  {int(metrics['fp']):4d}")
    print(f"       Fake   {int(metrics['fn']):4d}  {int(metrics['tp']):4d}")
    print(f"\n📈 Metrics:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1-Score:  {metrics['f1']:.4f}")
    print(f"   ROC-AUC:   {metrics['auc']:.4f}")
    print(f"   Real Acc:  {metrics['real_acc']:.4f}")
    print(f"   Fake Acc:  {metrics['fake_acc']:.4f}")

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    all_metrics = []
    
    pbar = tqdm(train_loader, desc="Training")
    
    for batch_idx, (images, labels, paths) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Compute metrics
        batch_metrics = compute_metrics(outputs, labels)
        all_metrics.append(batch_metrics)
        
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{batch_metrics["accuracy"]:.3f}',
            'F1': f'{batch_metrics["f1"]:.3f}'
        })
    
    # Compute average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    return total_loss / len(train_loader), avg_metrics

def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        
        for images, labels, _ in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Collect predictions and labels for full-set metrics
            probs = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:,1].cpu().numpy())  # Probability of fake class
            
            total_loss += loss.item()
            
            # Update progress bar with batch metrics
            batch_metrics = compute_metrics(outputs, labels)
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{batch_metrics["accuracy"]:.3f}',
                'Real': f'{batch_metrics["real_acc"]:.3f}',
                'Fake': f'{batch_metrics["fake_acc"]:.3f}'
            })
    
    # Compute metrics on entire validation set
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Basic metrics
    correct = (all_predictions == all_labels).sum()
    total = len(all_labels)
    accuracy = correct / total
    
    # Per-class accuracy
    real_mask = (all_labels == 0)
    fake_mask = (all_labels == 1)
    
    real_acc = 0.0
    fake_acc = 0.0
    if real_mask.sum() > 0:
        real_acc = (all_predictions[real_mask] == all_labels[real_mask]).mean()
    if fake_mask.sum() > 0:
        fake_acc = (all_predictions[fake_mask] == all_labels[fake_mask]).mean()
    
    # Confusion matrix
    tp = ((all_predictions == 1) & (all_labels == 1)).sum()
    tn = ((all_predictions == 0) & (all_labels == 0)).sum()
    fp = ((all_predictions == 1) & (all_labels == 0)).sum()
    fn = ((all_predictions == 0) & (all_labels == 1)).sum()
    
    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # ROC-AUC
    try:
        if len(np.unique(all_labels)) > 1:  # Only compute if both classes present
            auc = sklearn.metrics.roc_auc_score(all_labels, all_probs)
        else:
            auc = 0.5  # Random performance when only one class
    except Exception:
        auc = 0.5
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = recall  # Same as recall
    
    val_metrics = {
        'accuracy': accuracy,
        'real_acc': real_acc,
        'fake_acc': fake_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }
    
    return total_loss / len(val_loader), val_metrics

# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, save_path)

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """Load model checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        return 0, 0.0
    
    print(f"📂 Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch']
    best_val_f1 = checkpoint.get('best_val_f1', 0.0)
    
    print(f"✅ Resumed from epoch {start_epoch}")
    return start_epoch, best_val_f1

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def generate_final_plots():
    """Generate final training plots"""
    print("📊 Generating final training plots...")
    # This function can be implemented to create training curves
    # For now, we'll just print a message
    print("   📈 Training plots generation placeholder")

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    # Set reproducibility
    set_seed(42)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Model setup
    model = VanillaCNN(num_conv=5)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    # Mixed precision training
    scaler = GradScaler() if device.type == 'cuda' else None
    
    # Data loaders
    print("📊 Setting up data loaders...")
    train_loader = make_clean_loader('manifests/combined_train.csv', batch_size=32, is_train=True)
    val_loader = make_clean_loader('manifests/combined_val.csv', batch_size=32, is_train=False)
    
    # Checkpoint directory
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Training parameters
    num_epochs = 20
    patience = 5
    best_val_f1 = 0.0
    patience_counter = 0
    start_epoch = 0
    
    # Try to resume from checkpoint
    latest_checkpoint = checkpoint_dir / 'clean_latest.pth'
    if latest_checkpoint.exists():
        start_epoch, best_val_f1 = load_checkpoint(model, optimizer, scheduler, latest_checkpoint, device)
    
    print(f"\n🚀 Starting training from epoch {start_epoch + 1}")
    print(f"   Total epochs: {num_epochs}")
    print(f"   Early stopping patience: {patience}")
    print(f"   Best validation F1 so far: {best_val_f1:.4f}")
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Training
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        
        # Validation
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Train - Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.3f}, F1: {train_metrics['f1']:.3f}")
        print(f"   Val   - Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.3f}, F1: {val_metrics['f1']:.3f}")
        print(f"   Real Acc - Train: {train_metrics['real_acc']:.3f}, Val: {val_metrics['real_acc']:.3f}")
        print(f"   Fake Acc - Train: {train_metrics['fake_acc']:.3f}, Val: {val_metrics['fake_acc']:.3f}")
        
        # Print confusion matrix for every epoch
        print(f"\n📊 Epoch {epoch+1} Confusion Matrix:")
        print_confusion_matrix(val_metrics)
        
        # Save checkpoint
        checkpoint_path = checkpoint_dir / f'clean_epoch_{epoch+1}.pth'
        save_checkpoint(model, optimizer, scheduler, epoch+1, val_metrics, checkpoint_path)
        
        # Save best model by F1 (better for imbalanced data)
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_path = checkpoint_dir / 'clean_best.pth'
            save_checkpoint(model, optimizer, scheduler, epoch+1, val_metrics, best_path)
            print(f"🏆 New best model! Val F1: {best_val_f1:.3f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping after {patience} epochs without F1 improvement")
            break
        
        scheduler.step()
    
    print(f"\n🎉 Training completed! Best validation F1: {best_val_f1:.3f}")
    
    # Generate final plots and analysis
    print("\n📊 Generating final training analysis...")
    generate_final_plots()

if __name__ == '__main__':
    main() 