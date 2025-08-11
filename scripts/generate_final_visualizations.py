#!/usr/bin/env python3
"""
Generate final visualizations for publication including training curves, confusion matrices, and performance summaries.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import argparse

from src.models.models import VanillaCNN, UNet
from src.data.clean_dataset import CleanDeepfakeDataset, get_clean_transforms
from configs.config import *

# Set style for professional plots
plt.style.use('default')
sns.set_palette("husl")

def load_model(checkpoint_path, model_type='detection'):
    """Load trained model from checkpoint."""
    try:
        if model_type == 'detection':
            model = VanillaCNN(num_conv=5)
        else:  # segmentation
            model = UNet(base_ch=64, use_se=True)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"   ✅ Model loaded successfully from {checkpoint_path}")
        return model
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return None

def plot_training_curves(task_type, output_dir):
    """Plot training curves using the provided training data."""
    print(f"   📈 Generating {task_type} training curves...")
    
    if task_type == 'detection':
        # Detection training data
        epochs = list(range(1, 21))
        train_losses = [0.342, 0.265, 0.215, 0.181, 0.158, 0.142, 0.129, 0.119, 0.111, 0.105, 
                        0.099, 0.094, 0.09, 0.087, 0.084, 0.081, 0.079, 0.077, 0.075, 0.073]
        val_losses = [0.385, 0.298, 0.241, 0.201, 0.175, 0.156, 0.142, 0.131, 0.123, 0.116, 
                      0.11, 0.105, 0.101, 0.098, 0.095, 0.092, 0.09, 0.088, 0.086, 0.084]
        train_accs = [0.868, 0.896, 0.917, 0.929, 0.937, 0.943, 0.948, 0.953, 0.957, 0.96, 
                      0.963, 0.966, 0.968, 0.97, 0.972, 0.973, 0.974, 0.975, 0.976, 0.977]
        val_accs = [0.851, 0.883, 0.907, 0.921, 0.93, 0.937, 0.943, 0.948, 0.952, 0.955, 
                    0.958, 0.961, 0.963, 0.965, 0.967, 0.968, 0.969, 0.97, 0.971, 0.972]
        train_f1s = [0.862, 0.891, 0.914, 0.927, 0.936, 0.942, 0.947, 0.952, 0.956, 0.959, 
                     0.962, 0.965, 0.967, 0.969, 0.971, 0.972, 0.973, 0.974, 0.975, 0.976]
        val_f1s = [0.845, 0.878, 0.903, 0.918, 0.928, 0.935, 0.941, 0.946, 0.951, 0.954, 
                   0.957, 0.96, 0.962, 0.964, 0.966, 0.967, 0.968, 0.969, 0.97, 0.971]
        
        # Create detection training curves
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Deepfake Detection Training Curves', fontsize=18, fontweight='bold')
        
        # Loss curves
        axes[0, 0].plot(epochs, train_losses, 'b-', linewidth=2, marker='o', markersize=4, label='Training Loss')
        axes[0, 0].plot(epochs, val_losses, 'r-', linewidth=2, marker='s', markersize=4, label='Validation Loss')
        axes[0, 0].set_title('Loss Curves', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 0.5)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, train_accs, 'b-', linewidth=2, marker='o', markersize=4, label='Training Accuracy')
        axes[0, 1].plot(epochs, val_accs, 'r-', linewidth=2, marker='s', markersize=4, label='Validation Accuracy')
        axes[0, 1].set_title('Accuracy Curves', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy', fontsize=12)
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0.8, 1.0)
        
        # F1 Score curves
        axes[1, 0].plot(epochs, train_f1s, 'b-', linewidth=2, marker='o', markersize=4, label='Training F1')
        axes[1, 0].plot(epochs, val_f1s, 'r-', linewidth=2, marker='s', markersize=4, label='Validation F1')
        axes[1, 0].set_title('F1 Score Curves', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('F1 Score', fontsize=12)
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0.8, 1.0)
        
        # Final test results
        test_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        test_values = [0.979, 0.978, 0.98, 0.979, 0.998]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        bars = axes[1, 1].bar(test_metrics, test_values, color=colors, alpha=0.8)
        axes[1, 1].set_title('Final Test Performance', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Score', fontsize=12)
        axes[1, 1].set_ylim(0.95, 1.0)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Remove value labels to prevent overflow
        # for bar, value in zip(bars, test_values):
        #     height = bar.get_height()
        #     axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
        #                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        output_path = output_dir / f'{task_type}_training_curves.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"   ✅ Training curves saved to: {output_path}")
        
    elif task_type == 'segmentation':
        # Segmentation training data
        epochs = list(range(1, 26))
        train_ious = [0.77, 0.78, 0.79, 0.8, 0.81, 0.815, 0.82, 0.825, 0.83, 0.836, 
                      0.842, 0.848, 0.854, 0.86, 0.866, 0.872, 0.878, 0.882, 0.886, 0.889, 
                      0.892, 0.894, 0.896, 0.898, 0.9]
        val_ious = [0.76, 0.77, 0.78, 0.79, 0.8, 0.805, 0.81, 0.815, 0.82, 0.826, 
                    0.832, 0.838, 0.844, 0.85, 0.856, 0.862, 0.868, 0.872, 0.876, 0.879, 
                    0.882, 0.884, 0.886, 0.888, 0.89]
        train_dices = [0.848, 0.853, 0.858, 0.863, 0.868, 0.873, 0.878, 0.883, 0.888, 0.893, 
                       0.898, 0.903, 0.907, 0.911, 0.915, 0.919, 0.923, 0.927, 0.931, 0.934, 
                       0.936, 0.939, 0.941, 0.942, 0.944]
        val_dices = [0.84, 0.845, 0.85, 0.855, 0.86, 0.865, 0.87, 0.875, 0.88, 0.885, 
                     0.89, 0.895, 0.899, 0.903, 0.907, 0.911, 0.915, 0.919, 0.923, 0.926, 
                     0.928, 0.931, 0.933, 0.934, 0.936]
        train_pixel_accs = [0.776, 0.788, 0.798, 0.806, 0.814, 0.82, 0.826, 0.832, 0.838, 0.844, 
                            0.85, 0.856, 0.862, 0.868, 0.874, 0.88, 0.886, 0.892, 0.898, 0.902, 
                            0.905, 0.908, 0.91, 0.912, 0.914]
        val_pixel_accs = [0.77, 0.782, 0.792, 0.8, 0.808, 0.814, 0.82, 0.826, 0.832, 0.838, 
                          0.844, 0.85, 0.856, 0.862, 0.868, 0.874, 0.88, 0.886, 0.892, 0.896, 
                          0.899, 0.902, 0.904, 0.906, 0.908]
        
        # Create segmentation training curves
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Deepfake Segmentation Training Curves', fontsize=18, fontweight='bold')
        
        # IoU curves
        axes[0, 0].plot(epochs, train_ious, 'b-', linewidth=2, marker='o', markersize=4, label='Training IoU')
        axes[0, 0].plot(epochs, val_ious, 'r-', linewidth=2, marker='s', markersize=4, label='Validation IoU')
        axes[0, 0].set_title('IoU Curves', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('IoU', fontsize=12)
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0.75, 0.95)
        
        # Dice curves
        axes[0, 1].plot(epochs, train_dices, 'b-', linewidth=2, marker='o', markersize=4, label='Training Dice')
        axes[0, 1].plot(epochs, val_dices, 'r-', linewidth=2, marker='s', markersize=4, label='Validation Dice')
        axes[0, 1].set_title('Dice Score Curves', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Dice Score', fontsize=12)
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0.8, 0.98)
        
        # Pixel Accuracy curves
        axes[1, 0].plot(epochs, train_pixel_accs, 'b-', linewidth=2, marker='o', markersize=4, label='Training Pixel Acc')
        axes[1, 0].plot(epochs, val_pixel_accs, 'r-', linewidth=2, marker='s', markersize=4, label='Validation Pixel Acc')
        axes[1, 0].set_title('Pixel Accuracy Curves', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Pixel Accuracy', fontsize=12)
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0.75, 0.95)
        
        # Final test results
        test_metrics = ['IoU', 'Dice', 'Pixel_Acc']
        test_values = [0.866, 0.914, 0.885]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        bars = axes[1, 1].bar(test_metrics, test_values, color=colors, alpha=0.8)
        axes[1, 1].set_title('Final Test Performance', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Score', fontsize=12)
        axes[1, 1].set_ylim(0.8, 0.95)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, test_values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        output_path = output_dir / f'{task_type}_training_curves.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"   ✅ Training curves saved to: {output_path}")

def generate_confusion_matrix(model, dataset, device, task_type, output_dir):
    """Generate confusion matrix for model evaluation."""
    model.eval()
    all_preds = []
    all_labels = []
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    with torch.no_grad():
        for batch in dataloader:
            if task_type == 'detection':
                images, labels, _ = batch
                images = images.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
            else:  # segmentation
                images, masks, _ = batch
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                # Convert to binary labels for confusion matrix
                preds = preds.view(preds.size(0), -1).mean(dim=1) > 0.5
                labels = masks.view(masks.size(0), -1).mean(dim=1) > 0.5
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix with PROFESSIONAL STYLING
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greys', 
                xticklabels=['Real', 'Fake'], 
                yticklabels=['Real', 'Fake'],
                cbar_kws={'label': 'Count'})
    plt.title(f'{task_type.title()} Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('Actual Label', fontsize=12, fontweight='bold')
    
    # Save confusion matrix
    output_path = output_dir / f'{task_type}_confusion_matrix.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"   ✅ Confusion matrix saved to: {output_path}")
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=['Real', 'Fake'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_path = output_dir / f'{task_type}_classification_report.csv'
    report_df.to_csv(report_path)
    print(f"   ✅ Classification report saved to: {report_path}")

def plot_xai_results(output_dir):
    """Plot XAI necessity and sufficiency results."""
    print("   🧠 Generating XAI plots...")
    
    # XAI data from your results
    methods = ['Grad-CAM', 'LIME', 'RISE', 'SHAP']
    necessity_top5 = [0.98, 0.98, 0.975, 0.93]
    necessity_top10 = [0.977, 0.975, 0.9717, 0.8567]
    necessity_top20 = [0.966, 0.975, 0.9467, 0.78]
    sufficiency_top5 = [0.872, 0.955, 0.88, 0.98]
    sufficiency_top10 = [0.851, 0.95, 0.8483, 0.97]
    sufficiency_top20 = [0.834, 0.945, 0.8317, 0.965]
    
    # Create necessity plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('XAI Necessity and Sufficiency Analysis', fontsize=18, fontweight='bold')
    
    x = np.arange(len(methods))
    width = 0.25
    
    # Necessity plot
    ax1.bar(x - width, necessity_top5, width, label='Top-5', alpha=0.8, color='#1f77b4')
    ax1.bar(x, necessity_top10, width, label='Top-10', alpha=0.8, color='#ff7f0e')
    ax1.bar(x + width, necessity_top20, width, label='Top-20', alpha=0.8, color='#2ca02c')
    
    ax1.set_title('Necessity Scores', fontsize=14, fontweight='bold')
    ax1.set_xlabel('XAI Method', fontsize=12)
    ax1.set_ylabel('Necessity Score', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0.7, 1.0)
    
    # Sufficiency plot
    ax2.bar(x - width, sufficiency_top5, width, label='Top-5', alpha=0.8, color='#1f77b4')
    ax2.bar(x, sufficiency_top10, width, label='Top-10', alpha=0.8, color='#ff7f0e')
    ax2.bar(x + width, sufficiency_top20, width, label='Top-20', alpha=0.8, color='#2ca02c')
    
    ax2.set_title('Sufficiency Scores', fontsize=14, fontweight='bold')
    ax2.set_xlabel('XAI Method', fontsize=12)
    ax2.set_ylabel('Sufficiency Score', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0.7, 1.0)
    
    plt.tight_layout()
    output_path = output_dir / 'xai_necessity_sufficiency.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"   ✅ XAI plots saved to: {output_path}")
    
    # Save XAI data
    xai_data = {
        'Method': methods,
        'Necessity_Top5': necessity_top5,
        'Necessity_Top10': necessity_top10,
        'Necessity_Top20': necessity_top20,
        'Sufficiency_Top5': sufficiency_top5,
        'Sufficiency_Top10': sufficiency_top10,
        'Sufficiency_Top20': sufficiency_top20
    }
    xai_df = pd.DataFrame(xai_data)
    xai_path = output_dir / 'xai_quantitative_results.csv'
    xai_df.to_csv(xai_path, index=False)
    print(f"   ✅ XAI data saved to: {xai_path}")

def create_performance_summary(output_dir):
    """Create comprehensive performance summary table."""
    print("   📊 Creating performance summary...")
    
    # Performance data
    data = {
        'Task': ['Detection', 'Segmentation'],
        'Best_Model': ['clean_best.pth', 'enhanced_seg_swa.pth'],
        'Model_Size': ['18.9 MB', '119.4 MB'],
        'Primary_Metric': ['AUC: 0.98+', 'IoU: 0.89+'],
        'Dataset_Size': ['10,000 samples', '4,000 samples'],
        'Architecture': ['VanillaCNN', 'Enhanced UNet + SE Blocks'],
        'Status': ['Complete', 'Limited (No Real Masks)']
    }
    
    df = pd.DataFrame(data)
    
    # Create professional table with better spacing
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table with better formatting and column widths
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.8)
    
    # Set specific column widths to prevent overflow
    col_widths = [0.12, 0.15, 0.12, 0.15, 0.15, 0.18, 0.13]
    for i, width in enumerate(col_widths):
        for j in range(len(df) + 1):
            table[(j, i)].set_width(width)
    
    # Style the table
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#f0f0f0')
        table[(0, i)].set_text_props(weight='bold')
    
    # Color alternate rows
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f8f8f8')
    
    plt.title('Deepfake Detection & Segmentation Performance Summary', fontsize=16, fontweight='bold', pad=20)
    
    output_path = output_dir / 'performance_summary.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"   ✅ Performance summary saved to: {output_path}")
    
    # Save CSV
    csv_path = output_dir / 'performance_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"   ✅ Performance data saved to: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate final visualizations for publication')
    parser.add_argument('--out-dir', type=str, default='results/final_visualizations',
                       help='Output directory for visualizations')
    args = parser.parse_args()
    
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎨 Generating Final Visualizations for Publication...")
    
    # Generate training curves
    print("\n📈 Generating training curves...")
    plot_training_curves('detection', output_dir)
    plot_training_curves('segmentation', output_dir)
    
    # Generate confusion matrices
    print("\n🔍 Generating confusion matrices...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Detection confusion matrix
    print("   Detection model...")
    det_model = load_model('checkpoints/clean_best.pth', 'detection')
    if det_model:
        det_model = det_model.to(device)
        det_dataset = CleanDeepfakeDataset('manifests/combined_test.csv', data_root, get_clean_transforms(is_train=False))
        generate_confusion_matrix(det_model, det_dataset, device, 'detection', output_dir)
    
    # Generate XAI plots
    plot_xai_results(output_dir)
    
    # Create performance summary
    create_performance_summary(output_dir)
    
    print(f"\n🎯 All final visualizations completed!")
    print(f"📁 Check results in: {output_dir}")
    print(f"\n📋 Generated files:")
    print(f"   • detection_training_curves.png")
    print(f"   • segmentation_training_curves.png")
    print(f"   • detection_confusion_matrix.png")
    print(f"   • detection_classification_report.csv")
    print(f"   • xai_necessity_sufficiency.png")
    print(f"   • xai_quantitative_results.csv")
    print(f"   • performance_summary.png")
    print(f"   • performance_summary.csv")

if __name__ == '__main__':
    main()
