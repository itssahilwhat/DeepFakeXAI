#!/usr/bin/env python3

import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.append('.')

from src.core.config import Config
from src.preprocessing.dataset import get_dataloader
from src.detection.deepfake_models import GradCAMClassifier, PatchForensicsModel, AttentionModel
from src.xai.explainability import XAIMetrics, MaskVisualizer

class ModelEvaluator:
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        self.xai_metrics = XAIMetrics()
        self.visualizer = MaskVisualizer()
        
        # Load test data
        self.test_loader = get_dataloader('test', config, supervision_filter=None)
        
        # Initialize models
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load all trained models"""
        checkpoint_dir = self.config.CHECKPOINT_DIR
        
        # Load GradCAM model
        gradcam_path = os.path.join(checkpoint_dir, 'gradcam_best.pth')
        if os.path.exists(gradcam_path):
            self.models['gradcam'] = GradCAMClassifier()
            checkpoint = torch.load(gradcam_path, map_location=self.device)
            self.models['gradcam'].load_state_dict(checkpoint['model_state_dict'])
            self.models['gradcam'].to(self.device)
            self.models['gradcam'].eval()
            print(f"Loaded GradCAM model from {gradcam_path}")
        
        # Load Patch-Forensics model
        patch_path = os.path.join(checkpoint_dir, 'patch_forensics_best.pth')
        if os.path.exists(patch_path):
            self.models['patch'] = PatchForensicsModel()
            checkpoint = torch.load(patch_path, map_location=self.device)
            self.models['patch'].load_state_dict(checkpoint['model_state_dict'])
            self.models['patch'].to(self.device)
            self.models['patch'].eval()
            print(f"Loaded Patch-Forensics model from {patch_path}")
        
        # Load Attention model
        attention_path = os.path.join(checkpoint_dir, 'attention_best.pth')
        if os.path.exists(attention_path):
            self.models['attention'] = AttentionModel()
            checkpoint = torch.load(attention_path, map_location=self.device)
            self.models['attention'].load_state_dict(checkpoint['model_state_dict'])
            self.models['attention'].to(self.device)
            self.models['attention'].eval()
            print(f"Loaded Attention model from {attention_path}")
    
    def evaluate_classification(self, model_name, model):
        """Evaluate classification performance"""
        print(f"Evaluating {model_name} classification...")
        
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, masks, labels in tqdm(self.test_loader, desc=f"Evaluating {model_name}"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                if model_name == 'gradcam':
                    logits = model(images)
                else:
                    logits, _ = model(images)
                
                probs = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
        
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
        auc = roc_auc_score(all_labels, all_probs)
        cm = confusion_matrix(all_labels, all_predictions)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm
        }
    
    def evaluate_localization(self, model_name, model):
        """Evaluate localization performance"""
        print(f"Evaluating {model_name} localization...")
        
        all_pred_masks = []
        all_gt_masks = []
        
        with torch.no_grad():
            for images, masks, labels in tqdm(self.test_loader, desc=f"Evaluating {model_name} masks"):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                if model_name == 'gradcam':
                    # Generate GradCAM heatmaps
                    logits = model(images)
                    heatmaps = []
                    for i in range(images.size(0)):
                        heatmap = model.get_gradcam()
                        if heatmap is not None:
                            heatmaps.append(heatmap[i].cpu().numpy())
                        else:
                            heatmaps.append(np.zeros((224, 224)))
                    pred_masks = torch.tensor(heatmaps).unsqueeze(1)
                else:
                    # Get predicted masks
                    _, pred_masks = model(images)
                
                all_pred_masks.extend(pred_masks.cpu().numpy())
                all_gt_masks.extend(masks.cpu().numpy())
        
        # Calculate mask metrics
        mask_metrics = self.xai_metrics.compute_mask_metrics(
            np.array(all_pred_masks), np.array(all_gt_masks)
        )
        
        return mask_metrics
    
    def generate_visualizations(self, num_samples=20):
        """Generate visualization samples"""
        print("Generating visualizations...")
        
        sample_count = 0
        for images, masks, labels in self.test_loader:
            if sample_count >= num_samples:
                break
            
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Generate predictions for each model
            predictions = {}
            
            with torch.no_grad():
                for model_name, model in self.models.items():
                    if model_name == 'gradcam':
                        logits = model(images)
                        heatmaps = []
                        for i in range(images.size(0)):
                            heatmap = model.get_gradcam()
                            if heatmap is not None:
                                heatmaps.append(heatmap[i].cpu().numpy())
                            else:
                                heatmaps.append(np.zeros((224, 224)))
                        predictions[model_name] = torch.tensor(heatmaps).unsqueeze(1)
                    else:
                        _, pred_masks = model(images)
                        predictions[model_name] = pred_masks.cpu()
            
            # Save visualizations for each sample
            for i in range(min(images.size(0), num_samples - sample_count)):
                image = images[i]
                gt_mask = masks[i]
                
                # Create masks dictionary
                masks_dict = {
                    'Ground_Truth': gt_mask.cpu(),
                    **{f'{name}_Pred': pred[i] for name, pred in predictions.items()}
                }
                
                # Save comparison
                filename = f"sample_{sample_count + i:03d}_comparison.png"
                self.visualizer.save_comparison(image, masks_dict, filename)
                
                # Save individual overlays
                for name, mask in masks_dict.items():
                    if name != 'Ground_Truth':
                        overlay_filename = f"sample_{sample_count + i:03d}_{name.lower().replace('_', '_')}_overlay.png"
                        self.visualizer.save_overlay(image, mask, name, overlay_filename)
            
            sample_count += images.size(0)
    
    def run_evaluation(self):
        """Run comprehensive evaluation of all models"""
        print("Starting Comprehensive Evaluation")
        
        results = {}
        
        # Evaluate each model
        for model_name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"Evaluating {model_name.upper()} Model")
            print(f"{'='*50}")
            
            # Classification evaluation
            cls_results = self.evaluate_classification(model_name, model)
            
            # Localization evaluation
            loc_results = self.evaluate_localization(model_name, model)
            
            results[model_name] = {
                'classification': cls_results,
                'localization': loc_results
            }
            
            # Print results
            print(f"\n{model_name.upper()} Results:")
            print(f"Classification:")
            print(f"  Accuracy: {cls_results['accuracy']:.4f}")
            print(f"  Precision: {cls_results['precision']:.4f}")
            print(f"  Recall: {cls_results['recall']:.4f}")
            print(f"  F1: {cls_results['f1']:.4f}")
            print(f"  AUC: {cls_results['auc']:.4f}")
            
            print(f"Localization:")
            print(f"  IoU: {loc_results['iou']:.4f}")
            print(f"  Pixel Accuracy: {loc_results['pixel_accuracy']:.4f}")
            print(f"  Average Precision: {loc_results['average_precision']:.4f}")
            print(f"  Pixel AUC: {loc_results['pixel_auc']:.4f}")
            print(f"  Correlation: {loc_results['correlation']:.4f}")
        
        # Generate visualizations
        print(f"\n{'='*50}")
        print("Generating Visualizations")
        print(f"{'='*50}")
        self.generate_visualizations()
        
        # Save results
        self.save_results(results)
        
        return results
    
    def save_results(self, results):
        """Save evaluation results to CSV"""
        print("Saving results...")
        
        # Create results DataFrame
        rows = []
        for model_name, model_results in results.items():
            cls = model_results['classification']
            loc = model_results['localization']
            
            rows.append({
                'Model': model_name,
                'Accuracy': cls['accuracy'],
                'Precision': cls['precision'],
                'Recall': cls['recall'],
                'F1': cls['f1'],
                'AUC': cls['auc'],
                'IoU': loc['iou'],
                'Pixel_Accuracy': loc['pixel_accuracy'],
                'Average_Precision': loc['average_precision'],
                'Pixel_AUC': loc['pixel_auc'],
                'Correlation': loc['correlation']
            })
        
        df = pd.DataFrame(rows)
        
        # Save to CSV
        output_path = os.path.join(self.config.OUTPUT_DIR, 'comprehensive_evaluation_results.csv')
        df.to_csv(output_path, index=False)
        
        print(f"Results saved to {output_path}")
        
        # Print summary
        print(f"\n{'='*50}")
        print("SUMMARY")
        print(f"{'='*50}")
        print(df.to_string(index=False))

if __name__ == '__main__':
    config = Config()
    
    # Ensure output directory exists
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Run evaluation
    evaluator = ModelEvaluator(config)
    results = evaluator.run_evaluation() 