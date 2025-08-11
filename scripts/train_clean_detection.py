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
# No TensorBoard - removed to avoid file errors
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import random
import os
from datetime import datetime
import sklearn.metrics

from src.models.models import VanillaCNN
from src.data.clean_dataset import make_clean_loader, create_combined_clean_dataset

def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Don't use deterministic algorithms for CUDA (causes issues)
    # torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(seed)

def compute_metrics(outputs, labels):
    """Compute comprehensive classification metrics."""
    probs = torch.softmax(outputs, dim=1)
    predictions = torch.argmax(outputs, dim=1)
    
    # Basic metrics
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
    
    # Confusion matrix
    tp = ((predictions == 1) & (labels == 1)).sum().item()
    tn = ((predictions == 0) & (labels == 0)).sum().item()
    fp = ((predictions == 1) & (labels == 0)).sum().item()
    fn = ((predictions == 0) & (labels == 1)).sum().item()
    
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
    print(f"   Precision: {metrics['precision']:.3f}")
    print(f"      Recall: {metrics['recall']:.3f}")
    print(f"         F1: {metrics['f1']:.3f}")
    print(f"Specificity: {metrics['specificity']:.3f}")

def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_metrics = []
    
    pbar = tqdm(train_loader, desc="Training")
    for images, labels, _ in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Compute metrics
        metrics = compute_metrics(outputs, labels)
        all_metrics.append(metrics)
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{metrics["accuracy"]:.3f}',
            'Real': f'{metrics["real_acc"]:.3f}',
            'Fake': f'{metrics["fake_acc"]:.3f}'
        })
    
    # Average metrics
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

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }, save_path)
    print(f"💾 Saved checkpoint: {save_path}")

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load model checkpoint."""
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        metrics = checkpoint['metrics']
        print(f"📂 Loaded checkpoint: {checkpoint_path}")
        print(f"   Epoch: {epoch}")
        print(f"   Best F1: {metrics.get('f1', 'N/A'):.3f}")
        return epoch, metrics
    return 0, None

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint file."""
    checkpoint_files = list(checkpoint_dir.glob('clean_epoch_*.pth'))
    if not checkpoint_files:
        return None
    
    # Extract epoch numbers and find the latest
    latest_epoch = 0
    latest_file = None
    for file in checkpoint_files:
        try:
            epoch = int(file.stem.split('_')[-1])
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_file = file
        except ValueError:
            continue
    
    return latest_file

def generate_final_plots():
    """Generate final training plots and analysis."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from pathlib import Path
        
        print("   📈 Generating training curves...")
        
        # Create outputs directory
        output_dir = Path('outputs')
        output_dir.mkdir(exist_ok=True)
        
        # Load training history from checkpoints
        checkpoint_dir = Path('checkpoints')
        checkpoint_files = sorted(checkpoint_dir.glob('clean_epoch_*.pth'))
        
        if not checkpoint_files:
            print("   ⚠️  No checkpoints found for plotting")
            return
        
        # Extract metrics from checkpoints
        epochs = []
        val_accs = []
        val_f1s = []
        val_precisions = []
        val_recalls = []
        
        for checkpoint_file in checkpoint_files:
            try:
                checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
                epoch = checkpoint['epoch']
                metrics = checkpoint['metrics']
                
                epochs.append(epoch)
                val_accs.append(metrics.get('accuracy', 0))
                val_f1s.append(metrics.get('f1', 0))
                val_precisions.append(metrics.get('precision', 0))
                val_recalls.append(metrics.get('recall', 0))
            except Exception as e:
                print(f"   ⚠️  Error loading {checkpoint_file}: {e}")
                continue
        
        if not epochs:
            print("   ⚠️  No valid metrics found for plotting")
            return
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Deepfake Detection Training Results', fontsize=16, fontweight='bold')
        
        # Accuracy plot
        axes[0, 0].plot(epochs, val_accs, 'g-', linewidth=2, label='Validation Accuracy')
        axes[0, 0].set_title('Accuracy Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # F1 Score plot
        axes[0, 1].plot(epochs, val_f1s, 'r-', linewidth=2, label='Validation F1')
        axes[0, 1].set_title('F1 Score Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision/Recall plot
        axes[1, 0].plot(epochs, val_precisions, 'orange', linewidth=2, label='Precision')
        axes[1, 0].plot(epochs, val_recalls, 'purple', linewidth=2, label='Recall')
        axes[1, 0].set_title('Precision & Recall Over Time')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Final confusion matrix visualization
        print("   📊 Creating confusion matrix visualization...")
        # Get the best model's confusion matrix
        best_checkpoint = checkpoint_dir / 'clean_best.pth'
        if best_checkpoint.exists():
            checkpoint = torch.load(best_checkpoint, map_location='cpu', weights_only=False)
            metrics = checkpoint['metrics']
            
            # Create confusion matrix heatmap
            cm = np.array([
                [metrics['tn'], metrics['fp']],
                [metrics['fn'], metrics['tp']]
            ])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Real', 'Fake'], 
                       yticklabels=['Real', 'Fake'],
                       ax=axes[1, 1])
            axes[1, 1].set_title('Confusion Matrix (Best Model)')
            axes[1, 1].set_xlabel('Predicted')
            axes[1, 1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'training_results.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Training plots saved to: {output_dir / 'training_results.png'}")
        
        # Save metrics to CSV
        print("   📋 Saving final metrics report...")
        import pandas as pd
        metrics_df = pd.DataFrame({
            'epoch': epochs,
            'val_accuracy': val_accs,
            'val_f1': val_f1s,
            'val_precision': val_precisions,
            'val_recall': val_recalls
        })
        metrics_df.to_csv(output_dir / 'training_metrics.csv', index=False)
        print(f"   ✅ Training metrics saved to: {output_dir / 'training_metrics.csv'}")
        
        # Print final summary
        print(f"\n📊 Final Training Summary:")
        print(f"   Best F1 Score: {max(val_f1s):.3f} (Epoch {epochs[val_f1s.index(max(val_f1s))]})")
        print(f"   Best Accuracy: {max(val_accs):.3f} (Epoch {epochs[val_accs.index(max(val_accs))]})")
        print(f"   Final F1 Score: {val_f1s[-1]:.3f}")
        print(f"   Final Accuracy: {val_accs[-1]:.3f}")
        
        # Run final test evaluation
        print("   🧪 Running final test evaluation...")
        try:
            # Import and run test
            from scripts.test_final import main as test_main
            test_main()
            print("   ✅ Final test completed successfully!")
        except Exception as e:
            print(f"   ⚠️  Test evaluation failed: {e}")
        
        print("   📁 Results saved to outputs/ directory")
        print("   🎯 Training analysis complete!")
        
    except ImportError:
        print("   ⚠️  matplotlib/seaborn not available for plotting")
    except Exception as e:
        print(f"   ⚠️  Error generating plots: {e}")

def main():
    # Set reproducibility
    set_seed(42)
    
    # Configuration
    data_root = Path('data')  # Will handle both wacv_data and ff_processed
    manifest_dir = Path('manifests')
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Training parameters
    batch_size = 4  # Maximum safe size for RTX 3050 4GB with 512x512
    num_epochs = 20  # Increased from 10 to 20
    learning_rate = 5e-5  # Lower learning rate
    weight_decay = 5e-2   # Higher weight decay
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Starting Clean Training")
    print(f"   Device: {device}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Weight decay: {weight_decay}")
    
    # Create model
    model = VanillaCNN().to(device)
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load deduplicated manifests
    print("\n📊 Loading deduplicated datasets...")
    
    # Real datasets
    celeba_train = manifest_dir / 'celeba_train.csv'
    celeba_val = manifest_dir / 'celeba_val.csv'
    ffhq_train = manifest_dir / 'ffhq_train.csv'
    ffhq_val = manifest_dir / 'ffhq_val.csv'
    
    # Fake datasets (CelebA-HQ + FF++)
    celebahq_train = manifest_dir / 'celebahq_train.csv'
    celebahq_val = manifest_dir / 'celebahq_val.csv'
    
    # Create combined datasets (FF++ ignored)
    train_manifests = [celeba_train, ffhq_train, celebahq_train]
    val_manifests = [celeba_val, ffhq_val, celebahq_val]
    
    # Balance to reasonable size
    max_samples_per_class = 40000  # 40k real + 40k fake = 80k total (80%)
    # Validation will be 5k real + 5k fake = 10k total (10%)
    
    train_df, val_df = create_combined_clean_dataset(
        train_manifests, val_manifests, data_root, max_samples_per_class
    )
    
    # Class-weighted loss (give fake class more voice)
    num_real = (train_df['dataset'].isin(['celeba', 'ffhq'])).sum()
    num_fake = (~train_df['dataset'].isin(['celeba', 'ffhq'])).sum()
    weight_real = 1.0
    weight_fake = num_real / num_fake
    
    print(f"   Class weights - Real: {weight_real:.3f}, Fake: {weight_fake:.3f}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([weight_real, weight_fake]).float().to(device))
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler('cuda') if torch.cuda.is_available() else None
    
    # No TensorBoard - will generate final plots at the end
    writer = None
    
    # Save combined manifests
    train_manifest = manifest_dir / 'combined_train.csv'
    val_manifest = manifest_dir / 'combined_val.csv'
    train_df.to_csv(train_manifest, index=False)
    val_df.to_csv(val_manifest, index=False)
    
    # Create data loaders
    print("\n🔄 Creating data loaders...")
    train_loader = make_clean_loader(
        train_manifest, data_root, batch_size=batch_size, 
        is_train=True, balance=True
    )
    val_loader = make_clean_loader(
        val_manifest, data_root, batch_size=batch_size,
        is_train=False, balance=True  # BALANCED VALIDATION
    )
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Check for existing checkpoints and resume
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    start_epoch = 0
    best_val_f1 = 0.0
    
    if latest_checkpoint:
        print(f"\n🔄 Found existing checkpoint: {latest_checkpoint.name}")
        user_input = input("Resume from checkpoint? (y/n): ").lower().strip()
        if user_input == 'y':
            start_epoch, best_metrics = load_checkpoint(model, optimizer, scheduler, latest_checkpoint)
            if best_metrics:
                best_val_f1 = best_metrics.get('f1', 0.0)
            print(f"🔄 Resuming from epoch {start_epoch + 1}")
        else:
            print("🆕 Starting fresh training")
    else:
        print("🆕 No existing checkpoints found, starting fresh")
    
    # Training loop
    print("\n🎯 Starting training...")
    patience = 3
    patience_counter = 0
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\n📖 Epoch {epoch+1}/{num_epochs}")
        print("=" * 50)
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        # Store metrics for final plotting (no TensorBoard)
        if writer:
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
            writer.add_scalar('Accuracy/Val', val_metrics['accuracy'], epoch)
            writer.add_scalar('F1/Train', train_metrics['f1'], epoch)
            writer.add_scalar('F1/Val', val_metrics['f1'], epoch)
            writer.add_scalar('Precision/Val', val_metrics['precision'], epoch)
            writer.add_scalar('Recall/Val', val_metrics['recall'], epoch)
            writer.add_scalar('Specificity/Val', val_metrics['specificity'], epoch)
            writer.add_scalar('AUC/Val', val_metrics['auc'], epoch)
        
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
    
    if writer:
        writer.close()

if __name__ == '__main__':
    main() 