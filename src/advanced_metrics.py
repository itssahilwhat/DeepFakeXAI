import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from src.config import Config


class AdvancedMetrics:
    """Advanced metrics for comprehensive deepfake detection evaluation"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.targets = []
        self.masks = []
        self.pred_masks = []
    
    def update(self, pred, target, pred_mask=None, target_mask=None):
        """Update metrics with batch data"""
        # Classification predictions
        if pred.dim() == 1:  # Classification output
            self.predictions.append(pred.cpu().numpy())
            self.targets.append(target.cpu().numpy())
        
        # Segmentation masks
        if pred_mask is not None and target_mask is not None:
            self.pred_masks.append(pred_mask.cpu().numpy())
            self.masks.append(target_mask.cpu().numpy())
    
    def compute(self):
        """Compute all advanced metrics"""
        metrics = {}
        
        if len(self.predictions) > 0:
            # Classification metrics
            preds = np.concatenate(self.predictions)
            targets = np.concatenate(self.targets)
            
            # ROC AUC
            try:
                metrics['roc_auc'] = roc_auc_score(targets, preds)
            except:
                metrics['roc_auc'] = 0.0
            
            # Average Precision
            try:
                metrics['avg_precision'] = average_precision_score(targets, preds)
            except:
                metrics['avg_precision'] = 0.0
            
            # EER (Equal Error Rate)
            metrics['eer'] = self._compute_eer(preds, targets)
            
            # HTER (Half Total Error Rate)
            metrics['hter'] = self._compute_hter(preds, targets)
        
        if len(self.pred_masks) > 0:
            # Segmentation metrics
            pred_masks = np.concatenate(self.pred_masks)
            target_masks = np.concatenate(self.masks)
            
            # Dice coefficient
            metrics['dice'] = self._compute_dice(pred_masks, target_masks)
            
            # IoU (Jaccard)
            metrics['iou'] = self._compute_iou(pred_masks, target_masks)
            
            # Boundary accuracy
            metrics['boundary_acc'] = self._compute_boundary_accuracy(pred_masks, target_masks)
            
            # Hausdorff distance
            metrics['hausdorff'] = self._compute_hausdorff_distance(pred_masks, target_masks)
        
        return metrics
    
    def _compute_eer(self, preds, targets, num_thresholds=1000):
        """Compute Equal Error Rate"""
        thresholds = np.linspace(0, 1, num_thresholds)
        fprs = []
        fnrs = []
        
        for thresh in thresholds:
            pred_labels = (preds > thresh).astype(int)
            tp = np.sum((pred_labels == 1) & (targets == 1))
            tn = np.sum((pred_labels == 0) & (targets == 0))
            fp = np.sum((pred_labels == 1) & (targets == 0))
            fn = np.sum((pred_labels == 0) & (targets == 1))
            
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            fprs.append(fpr)
            fnrs.append(fnr)
        
        # Find threshold where FPR ≈ FNR
        diff = np.abs(np.array(fprs) - np.array(fnrs))
        eer_idx = np.argmin(diff)
        
        return (fprs[eer_idx] + fnrs[eer_idx]) / 2
    
    def _compute_hter(self, preds, targets, threshold=0.5):
        """Compute Half Total Error Rate"""
        pred_labels = (preds > threshold).astype(int)
        tp = np.sum((pred_labels == 1) & (targets == 1))
        tn = np.sum((pred_labels == 0) & (targets == 0))
        fp = np.sum((pred_labels == 1) & (targets == 0))
        fn = np.sum((pred_labels == 0) & (targets == 1))
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return (fpr + fnr) / 2
    
    def _compute_dice(self, pred_masks, target_masks, threshold=0.5):
        """Compute Dice coefficient"""
        pred_binary = (pred_masks > threshold).astype(float)
        intersection = np.sum(pred_binary * target_masks)
        union = np.sum(pred_binary) + np.sum(target_masks)
        return (2 * intersection) / (union + 1e-6)
    
    def _compute_iou(self, pred_masks, target_masks, threshold=0.5):
        """Compute IoU (Jaccard index)"""
        pred_binary = (pred_masks > threshold).astype(float)
        intersection = np.sum(pred_binary * target_masks)
        union = np.sum(pred_binary + target_masks - pred_binary * target_masks)
        return intersection / (union + 1e-6)
    
    def _compute_boundary_accuracy(self, pred_masks, target_masks, threshold=0.5):
        """Compute boundary accuracy"""
        pred_binary = (pred_masks > threshold).astype(float)
        
        # Extract boundaries using morphological operations
        from scipy import ndimage
        
        pred_boundary = ndimage.binary_erosion(pred_binary) != pred_binary
        target_boundary = ndimage.binary_erosion(target_masks) != target_masks
        
        intersection = np.sum(pred_boundary & target_boundary)
        union = np.sum(pred_boundary | target_boundary)
        
        return intersection / (union + 1e-6)
    
    def _compute_hausdorff_distance(self, pred_masks, target_masks, threshold=0.5):
        """Compute Hausdorff distance (simplified)"""
        pred_binary = (pred_masks > threshold).astype(float)
        
        # Simplified Hausdorff distance using centroid distance
        try:
            from scipy.spatial.distance import cdist
            from scipy.ndimage import center_of_mass
            
            pred_centroid = center_of_mass(pred_binary)
            target_centroid = center_of_mass(target_masks)
            
            if pred_centroid[0] is not None and target_centroid[0] is not None:
                distance = np.sqrt((pred_centroid[0] - target_centroid[0])**2 + 
                                 (pred_centroid[1] - target_centroid[1])**2)
                return distance
            else:
                return float('inf')
        except:
            return float('inf')


class TemporalMetrics:
    """Metrics for temporal consistency evaluation"""
    
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.reset()
    
    def reset(self):
        self.frame_predictions = []
        self.temporal_consistency_scores = []
    
    def update(self, predictions):
        """Update with frame predictions"""
        self.frame_predictions.append(predictions.cpu().numpy())
        
        # Keep only recent frames
        if len(self.frame_predictions) > self.window_size:
            self.frame_predictions.pop(0)
    
    def compute_temporal_consistency(self):
        """Compute temporal consistency score"""
        if len(self.frame_predictions) < 2:
            return 0.0
        
        consistency_scores = []
        for i in range(1, len(self.frame_predictions)):
            prev_pred = self.frame_predictions[i-1]
            curr_pred = self.frame_predictions[i]
            
            # Compute correlation between consecutive frames
            correlation = np.corrcoef(prev_pred.flatten(), curr_pred.flatten())[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            consistency_scores.append(correlation)
        
        return np.mean(consistency_scores)
    
    def compute(self):
        """Compute all temporal metrics"""
        return {
            'temporal_consistency': self.compute_temporal_consistency(),
            'frame_count': len(self.frame_predictions)
        }


class RobustnessMetrics:
    """Metrics for robustness evaluation"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.clean_metrics = {}
        self.perturbed_metrics = {}
    
    def update_clean(self, metrics):
        """Update clean performance metrics"""
        self.clean_metrics = metrics
    
    def update_perturbed(self, perturbation_type, metrics):
        """Update perturbed performance metrics"""
        self.perturbed_metrics[perturbation_type] = metrics
    
    def compute_robustness_score(self):
        """Compute overall robustness score"""
        if not self.clean_metrics or not self.perturbed_metrics:
            return 0.0
        
        robustness_scores = []
        
        for pert_type, pert_metrics in self.perturbed_metrics.items():
            # Compute degradation for each metric
            degradations = []
            for metric in ['accuracy', 'dice', 'iou', 'f1']:
                if metric in self.clean_metrics and metric in pert_metrics:
                    clean_val = self.clean_metrics[metric]
                    pert_val = pert_metrics[metric]
                    if clean_val > 0:
                        degradation = 1 - (pert_val / clean_val)
                        degradations.append(max(0, degradation))
            
            if degradations:
                robustness_scores.append(1 - np.mean(degradations))
        
        return np.mean(robustness_scores) if robustness_scores else 0.0
    
    def compute(self):
        """Compute all robustness metrics"""
        return {
            'robustness_score': self.compute_robustness_score(),
            'clean_metrics': self.clean_metrics,
            'perturbed_metrics': self.perturbed_metrics
        } 