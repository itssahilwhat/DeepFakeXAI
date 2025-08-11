#!/usr/bin/env python3
"""
Comprehensive XAI Analysis and Comparison Script
Generates explanations using multiple XAI methods (Grad-CAM++, RISE, SHAP, LIME)
and creates comparison visualizations for deepfake detection and segmentation.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent / '..'))

from src.models.models import VanillaCNN, UNet
from src.data.clean_dataset import CleanDeepfakeDataset, get_clean_transforms
from src.explainability.explain import explain_gradcam, explain_lime, explain_rise, explain_shap, explain_gradcam_plus_plus
from configs.config import data_root

class ComprehensiveXAIAnalyzer:
    """Comprehensive XAI analyzer for multiple explanation methods."""
    
    def __init__(self, model, device, task_type='detection'):
        self.model = model
        self.device = device
        self.task_type = task_type
        self.model.eval()
        
        # Results storage
        self.results = {}
        
    def compute_explanation_metrics(self, images, labels, analysis_type='comprehensive'):
        """Compute explanation metrics for given images using multiple XAI methods."""
        results = {}
        
        # Define available XAI methods
        xai_methods = {
            'Grad-CAM': 'gradcam',
            'LIME': 'lime'
        }
        
        for method_name, method_type in xai_methods.items():
            print(f"         🔍 Computing {method_name} explanations...")
            
            method_results = []
            
            for i in range(images.size(0)):
                # Get a fresh copy of the image for each iteration
                image = images[i:i+1].clone()
                label = labels[i:i+1] if self.task_type == 'detection' else labels[i:i+1]
                
                try:
                    # Get explanation using available functions
                    if method_type == 'gradcam':
                        # Debug: print original shape
                        print(f"            🔍 Original image shape: {image.shape}")
                        
                        # Handle multiple possible shapes: [1, 1, 3, 224, 224] -> [1, 3, 224, 224]
                        # or [1, 3, 224, 224] -> [1, 3, 224, 224]
                        while image.dim() > 4:
                            image = image.squeeze(1)
                            print(f"            🔍 After squeeze(1): {image.shape}")
                        
                        if image.dim() == 3:  # [3, 224, 224] -> [1, 3, 224, 224]
                            image = image.unsqueeze(0)
                            print(f"            🔍 After unsqueeze(0): {image.shape}")
                        
                        # Final shape check
                        if image.dim() != 4:
                            print(f"            ❌ Invalid shape after processing: {image.shape}")
                            continue
                            
                        image = image.to(self.device)
                        # Ensure label is on the same device as the model
                        label_tensor = torch.tensor([label.item()], device=self.device)
                        explanation = explain_gradcam(self.model, image, label_tensor)
                    elif method_type == 'lime':
                        # Convert tensor to numpy for LIME
                        img_np = image.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
                        img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
                        img_np = np.clip(img_np, 0, 1)
                        
                        def predict_fn(images):
                            # Convert numpy back to tensor
                            img_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float()
                            img_tensor = (img_tensor - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                            img_tensor = img_tensor.to(self.device)
                            with torch.no_grad():
                                outputs = self.model(img_tensor)
                                if self.task_type == 'detection':
                                    probs = torch.softmax(outputs, dim=1)
                                else:
                                    probs = torch.sigmoid(outputs)
                            return probs.detach().cpu().numpy()
                        
                        explanation = explain_lime(predict_fn, img_np)
                    
                    # Compute metrics for this explanation
                    metrics = self._compute_explanation_metrics(explanation, label, image)
                    method_results.append(metrics)
                    
                except Exception as e:
                    print(f"            ❌ Error processing image {i} with {method_name}: {e}")
                    continue
            
            if method_results:
                results[method_name] = method_results
        
        return results
    
    def _compute_explanation_metrics(self, explanation, label, image):
        """Compute metrics for a single explanation."""
        # Basic metrics - you can expand these
        metrics = {
            'explanation_mean': float(np.mean(explanation)),
            'explanation_std': float(np.std(explanation)),
            'explanation_max': float(np.max(explanation)),
            'explanation_min': float(np.min(explanation)),
            'label': int(label.item())
        }
        return metrics
    
    def analyze_comprehensive_xai(self, dataloader, output_dir):
        """Run comprehensive XAI analysis with multiple methods."""
        print(f"🔬 Running comprehensive XAI analysis...")
        
        all_results = []
        total_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            if len(batch) >= 2:
                images, labels = batch[0], batch[1]
            else:
                # Handle case where batch might be just images
                images = batch[0]
                labels = torch.zeros(images.size(0), dtype=torch.long)
            
            print(f"   📊 Processing batch {batch_idx + 1}/{total_batches} ({images.size(0)} images)")
            
            # Compute explanations for this batch
            batch_results = self.compute_explanation_metrics(images, labels)
            all_results.append(batch_results)
            
            # Limit to first few batches for demonstration
            if batch_idx >= 2:  # Process first 3 batches
                break
        
        # Save results
        self._save_results(output_dir, all_results)
        
        # Generate plots
        self._plot_results(output_dir, all_results)
        
        print(f"✅ Comprehensive XAI analysis completed!")
    
    def _save_results(self, output_dir, results):
        """Save analysis results to CSV."""
        output_path = Path(output_dir) / 'comprehensive_xai_results.csv'
        
        # Flatten results for CSV
        flat_results = []
        for batch_results in results:
            for method_name, method_results in batch_results.items():
                for result in method_results:
                    result['method'] = method_name
                    flat_results.append(result)
        
        if flat_results:
            df = pd.DataFrame(flat_results)
            df.to_csv(output_path, index=False)
            print(f"📊 Saved results to: {output_path}")
        else:
            print("⚠️  No results to save")
    
    def _plot_results(self, output_dir, results):
        """Generate plots from analysis results."""
        try:
            # Simple summary plots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Comprehensive XAI Analysis Results', fontsize=16, fontweight='bold')
            
            # Flatten results for plotting
            flat_results = []
            for batch_results in results:
                for method_name, method_results in batch_results.items():
                    for result in method_results:
                        result['method'] = method_name
                        flat_results.append(result)
            
            if not flat_results:
                print("⚠️  No results to plot")
                return
            
            df = pd.DataFrame(flat_results)
            
            # Plot 1: Explanation statistics by method
            methods = df['method'].unique()
            means = [df[df['method'] == method]['explanation_mean'].mean() for method in methods]
            axes[0, 0].bar(methods, means)
            axes[0, 0].set_title('Average Explanation Values by Method')
            axes[0, 0].set_ylabel('Mean Explanation Value')
            
            # Plot 2: Explanation distribution
            for method in methods:
                method_data = df[df['method'] == method]['explanation_mean']
                axes[0, 1].hist(method_data, alpha=0.7, label=method, bins=20)
            axes[0, 1].set_title('Explanation Value Distribution')
            axes[0, 1].set_xlabel('Explanation Value')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            
            # Plot 3: Method comparison
            axes[1, 0].boxplot([df[df['method'] == method]['explanation_std'] for method in methods], labels=methods)
            axes[1, 0].set_title('Explanation Consistency by Method')
            axes[1, 0].set_ylabel('Standard Deviation')
            
            # Plot 4: Label distribution
            label_counts = df['label'].value_counts()
            axes[1, 1].pie(label_counts.values, labels=['Real' if i == 0 else 'Fake' for i in label_counts.index], autopct='%1.1f%%')
            axes[1, 1].set_title('Sample Distribution by Label')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = Path(output_dir) / 'comprehensive_xai_analysis_plots.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📈 Saved analysis plots to: {plot_path}")
            
            plt.close()
            
        except Exception as e:
            print(f"⚠️  Error generating plots: {e}")

def load_model(checkpoint_path, model_type='detection'):
    """Load pre-trained model."""
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    try:
        if model_type == 'detection':
            model = VanillaCNN()
        else:  # segmentation
            model = UNet()
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded successfully from {checkpoint_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def generate_xai_comparison_visualization(model, dataloader, output_dir, num_samples=4):
    """
    Generate comprehensive XAI comparison visualization showing all methods side by side.
    This creates the grid layout similar to the image you shared.
    """
    print(f"🎨 Generating XAI comparison visualization for {num_samples} samples...")
    
    model.eval()
    device = next(model.parameters()).device
    
    # Get sample images
    sample_images = []
    sample_labels = []
    
    for batch in dataloader:
        if len(sample_images) >= num_samples:
            break
        images, labels = batch[0], batch[1]
        for i in range(min(images.size(0), num_samples - len(sample_images))):
            sample_images.append(images[i])
            sample_labels.append(labels[i])
    
    # Create the grid visualization
    fig, axes = plt.subplots(5, num_samples, figsize=(4*num_samples, 20))
    fig.suptitle('XAI Methods Comparison', fontsize=20, fontweight='bold')
    
    # Row labels
    row_labels = ['Input', 'Grad-CAM++', 'RISE', 'SHAP', 'LIME']
    
    for row, (method, label) in enumerate(zip(row_labels, range(5))):
        for col in range(num_samples):
            ax = axes[row, col] if num_samples > 1 else axes[row]
            
            if row == 0:  # Input images
                # Convert tensor to numpy and denormalize
                img_np = sample_images[col].detach().cpu().permute(1, 2, 0).numpy()
                img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
                img_np = np.clip(img_np, 0, 1)
                
                ax.imshow(img_np)
                ax.set_title(f'Image {col+1}', fontsize=12)
                ax.axis('off')
                
            else:  # XAI methods
                img_tensor = sample_images[col].unsqueeze(0).to(device)
                label_tensor = torch.tensor([sample_labels[col].item()], device=device)
                
                try:
                    if method == 'Grad-CAM++':
                        heatmap = explain_gradcam_plus_plus(model, img_tensor, label_tensor)
                        # Overlay heatmap on original image
                        img_np = sample_images[col].detach().cpu().permute(1, 2, 0).numpy()
                        img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
                        img_np = np.clip(img_np, 0, 1)
                        
                        ax.imshow(img_np)
                        ax.imshow(heatmap, cmap='jet', alpha=0.6)
                        ax.set_title('Grad-CAM++', fontsize=12)
                        
                    elif method == 'RISE':
                        heatmap = explain_rise(model, img_tensor, label_tensor.item())
                        # Overlay heatmap on original image
                        img_np = sample_images[col].detach().cpu().permute(1, 2, 0).numpy()
                        img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
                        img_np = np.clip(img_np, 0, 1)
                        
                        ax.imshow(img_np)
                        ax.imshow(heatmap, cmap='jet', alpha=0.6)
                        ax.set_title('RISE', fontsize=12)
                        
                    elif method == 'SHAP':
                        heatmap = explain_shap(model, img_tensor, label_tensor.item())
                        if heatmap is not None:
                            # Overlay heatmap on original image
                            img_np = sample_images[col].detach().cpu().permute(1, 2, 0).numpy()
                            img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
                            img_np = np.clip(img_np, 0, 1)
                            
                            ax.imshow(img_np)
                            ax.imshow(heatmap, cmap='jet', alpha=0.6)
                            ax.set_title('SHAP', fontsize=12)
                        else:
                            ax.text(0.5, 0.5, 'SHAP\nNot Available', ha='center', va='center', transform=ax.transAxes)
                            ax.set_title('SHAP', fontsize=12)
                            
                    elif method == 'LIME':
                        # Convert tensor to numpy for LIME
                        img_np = sample_images[col].detach().cpu().permute(1, 2, 0).numpy()
                        img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
                        img_np = np.clip(img_np, 0, 1)
                        
                        def predict_fn(images):
                            # Convert numpy back to tensor
                            img_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float()
                            img_tensor = (img_tensor - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                            img_tensor = img_tensor.to(device)
                            with torch.no_grad():
                                outputs = model(img_tensor.unsqueeze(0))
                                probs = torch.softmax(outputs, dim=1)
                            return probs.detach().cpu().numpy()
                        
                        lime_mask = explain_lime(predict_fn, img_np)
                        
                        # Overlay LIME mask on original image
                        ax.imshow(img_np)
                        ax.imshow(lime_mask, cmap='YlOrRd', alpha=0.7)
                        ax.set_title('LIME', fontsize=12)
                    
                    ax.axis('off')
                    
                except Exception as e:
                    print(f"Error generating {method} for sample {col}: {e}")
                    ax.text(0.5, 0.5, f'{method}\nError', ha='center', va='center', transform=ax.transAxes, color='red')
                    ax.set_title(method, fontsize=12)
                    ax.axis('off')
    
    # Add row labels
    for row, label in enumerate(row_labels):
        if num_samples > 1:
            axes[row, 0].set_ylabel(label, fontsize=14, fontweight='bold', rotation=90)
        else:
            axes[row].set_ylabel(label, fontsize=14, fontweight='bold', rotation=90)
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = Path(output_dir) / 'xai_comparison_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"🎨 Saved XAI comparison visualization to: {output_path}")
    
    plt.close()
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Comprehensive XAI Analysis and Comparison')
    parser.add_argument('--task', type=str, choices=['detection', 'segmentation'], required=True,
                       help='Task type: detection or segmentation')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to pre-trained model checkpoint')
    parser.add_argument('--test-manifest', type=str, required=True,
                       help='Path to test manifest CSV')
    parser.add_argument('--out-dir', type=str, default='results/comprehensive_xai',
                       help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for analysis')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Load model
    print(f"📥 Loading {args.task} model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.task)
    if model is None:
        return
    
    model = model.to(device)
    
    # Load test dataset
    print(f"📊 Loading test data from {args.test_manifest}...")
    test_dataset = CleanDeepfakeDataset(args.test_manifest, data_root, get_clean_transforms(is_train=False))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    print(f"✅ Test samples: {len(test_dataset)}")
    
    # Initialize analyzer
    analyzer = ComprehensiveXAIAnalyzer(model, device, args.task)
    
    # Run analysis
    print(f"\n🔬 Starting comprehensive XAI analysis for {args.task}...")
    analyzer.analyze_comprehensive_xai(test_loader, output_dir)
    
    # Generate comprehensive XAI comparison visualization
    print(f"\n🎨 Generating XAI comparison visualization...")
    try:
        generate_xai_comparison_visualization(model, test_loader, output_dir, num_samples=4)
        print(f"✅ XAI comparison visualization generated successfully!")
    except Exception as e:
        print(f"⚠️  Error generating XAI comparison visualization: {e}")
    
    print(f"\n🎯 Comprehensive XAI analysis completed!")
    print(f"📁 Check results in: {output_dir}")

if __name__ == '__main__':
    main()
