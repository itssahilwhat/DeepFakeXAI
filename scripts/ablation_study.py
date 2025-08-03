#!/usr/bin/env python3

"""
Comprehensive ablation study script to compare detection performance across different models.
Compares GradCAM vs Patch vs Attention vs different backbones.
"""

import os
import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append('.')

from src.core.config import Config
from src.evaluation.metrics import ClassificationMetrics
from src.preprocessing.dataset import get_dataloader
from src.detection.deepfake_models import (
    GradCAMClassifier, PatchForensicsModel, AttentionModel, 
    EfficientNetClassifier, ResNetSE, MobileNetV3Classifier
)

class AblationStudy:
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        self.metrics = ClassificationMetrics()
        
        self.test_loader = get_dataloader('test', config, supervision_filter=None)
        
        self.model_configs = [
            {'name': 'GradCAM-Xception', 'model': GradCAMClassifier, 'params': {}},
            {'name': 'Patch-Forensics', 'model': PatchForensicsModel, 'params': {}},
            {'name': 'Attention-Xception', 'model': AttentionModel, 'params': {}},
            {'name': 'EfficientNet-B0', 'model': EfficientNetClassifier, 'params': {'backbone': 'efficientnet_b0'}},
            {'name': 'EfficientNet-B1', 'model': EfficientNetClassifier, 'params': {'backbone': 'efficientnet_b1'}},
            {'name': 'EfficientNet-B2', 'model': EfficientNetClassifier, 'params': {'backbone': 'efficientnet_b2'}},
            {'name': 'ResNet34-SE', 'model': ResNetSE, 'params': {'backbone': 'resnet34'}},
            {'name': 'ResNet50-SE', 'model': ResNetSE, 'params': {'backbone': 'resnet50'}},
            {'name': 'MobileNetV3', 'model': MobileNetV3Classifier, 'params': {}},
        ]
        
        self.results = []
        
    def load_model(self, model_config, checkpoint_path):
        try:
            model = model_config['model'](**model_config['params'])
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Failed to load {model_config['name']}: {e}")
            return None
    
    def evaluate_model(self, model, model_name):
        print(f"Evaluating {model_name}...")
        
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                images, masks, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                if hasattr(model, 'forward'):
                    outputs = model(images)
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                else:
                    continue
                
                probs = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        metrics = self.metrics.calculate_metrics(all_predictions, all_labels, all_probs)
        metrics['model_name'] = model_name
        metrics['num_params'] = sum(p.numel() for p in model.parameters())
        
        return metrics
    
    def run_ablation_study(self):
        print("Running Comprehensive Ablation Study")
        print("="*50)
        
        checkpoint_dir = Path(self.config.CHECKPOINT_DIR)
        
        for model_config in self.model_configs:
            model_name = model_config['name']
            
            possible_checkpoints = [
                checkpoint_dir / f"{model_name.lower().replace('-', '_')}_best.pth",
                checkpoint_dir / f"{model_name.lower().replace('-', '_').replace('_', '')}_best.pth",
                checkpoint_dir / f"{model_name.lower().split('-')[0]}_best.pth",
            ]
            
            model = None
            for checkpoint_path in possible_checkpoints:
                if checkpoint_path.exists():
                    model = self.load_model(model_config, checkpoint_path)
                    if model is not None:
                        break
            
            if model is not None:
                metrics = self.evaluate_model(model, model_name)
                self.results.append(metrics)
                print(f"{model_name}: AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}")
            else:
                print(f"{model_name}: No checkpoint found")
        
        return self.results
    
    def analyze_results(self):
        if not self.results:
            print("No results to analyze!")
            return
        
        df = pd.DataFrame(self.results)
        df = df.sort_values('f1', ascending=False)
        
        print("\nAblation Study Results")
        print("="*50)
        print(df[['model_name', 'accuracy', 'precision', 'recall', 'f1', 'auc', 'num_params']].to_string(index=False))
        
        self._create_visualizations(df)
        
        output_dir = Path('outputs/ablation_study')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_dir / 'ablation_results.csv', index=False)
        print(f"\nResults saved to: {output_dir}")
        
        return df
    
    def _create_visualizations(self, df):
        output_dir = Path('outputs/ablation_study')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        axes[0, 0].barh(df['model_name'], df['f1'], color='skyblue')
        axes[0, 0].set_title('F1 Score Comparison')
        axes[0, 0].set_xlabel('F1 Score')
        axes[0, 0].set_xlim(0, 1)
        
        axes[0, 1].barh(df['model_name'], df['auc'], color='lightgreen')
        axes[0, 1].set_title('AUC Score Comparison')
        axes[0, 1].set_xlabel('AUC Score')
        axes[0, 1].set_xlim(0, 1)
        
        scatter = axes[1, 0].scatter(df['num_params'] / 1e6, df['f1'], 
                                   s=100, alpha=0.7, c=df['auc'], cmap='viridis')
        axes[1, 0].set_xlabel('Model Parameters (M)')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('Model Size vs Performance')
        plt.colorbar(scatter, ax=axes[1, 0], label='AUC Score')
        
        axes[1, 1].scatter(df['recall'], df['precision'], s=100, alpha=0.7)
        for i, model_name in enumerate(df['model_name']):
            axes[1, 1].annotate(model_name, (df['recall'].iloc[i], df['precision'].iloc[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].set_title('Precision-Recall Trade-off')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'ablation_visualizations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to: {output_dir / 'ablation_visualizations.png'}")
    
    def generate_recommendations(self, df):
        print("\nRecommendations")
        print("="*50)
        
        best_overall = df.loc[df['f1'].idxmax()]
        print(f"Best Overall: {best_overall['model_name']} (F1: {best_overall['f1']:.4f})")
        
        lightweight_models = df[df['num_params'] < 10e6]
        if not lightweight_models.empty:
            best_lightweight = lightweight_models.loc[lightweight_models['f1'].idxmax()]
            print(f"Best Lightweight: {best_lightweight['model_name']} (F1: {best_lightweight['f1']:.4f}, Params: {best_lightweight['num_params']/1e6:.1f}M)")
        
        best_precision = df.loc[df['precision'].idxmax()]
        print(f"Best Precision: {best_precision['model_name']} (Precision: {best_precision['precision']:.4f})")
        
        best_recall = df.loc[df['recall'].idxmax()]
        print(f"Best Recall: {best_recall['model_name']} (Recall: {best_recall['recall']:.4f})")
        
        print(f"\nUse Case Recommendations:")
        print(f"  Production Deployment: {best_overall['model_name']}")
        if not lightweight_models.empty:
            print(f"  Mobile/Edge: {best_lightweight['model_name']}")
        print(f"  High Precision Required: {best_precision['model_name']}")
        print(f"  High Recall Required: {best_recall['model_name']}")

def main():
    config = Config()
    
    study = AblationStudy(config)
    results = study.run_ablation_study()
    
    if results:
        df = study.analyze_results()
        study.generate_recommendations(df)
    else:
        print("No models found to evaluate. Please train some models first.")

if __name__ == "__main__":
    main() 