import torch
import torch.nn.functional as F
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
import os

class XAIMetrics:
    """XAI metrics for mask evaluation and explainability"""
    
    def __init__(self):
        pass
    
    def compute_iou(self, pred_mask, gt_mask, threshold=0.5):
        """Compute Intersection over Union"""
        pred_binary = (pred_mask > threshold).astype(np.float32)
        gt_binary = (gt_mask > 0.5).astype(np.float32)
        
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()
        
        return intersection / (union + 1e-8)
    
    def compute_pixel_accuracy(self, pred_mask, gt_mask, threshold=0.5):
        """Compute Pixel Binary Classification Accuracy"""
        pred_binary = (pred_mask > threshold).astype(np.float32)
        gt_binary = (gt_mask > 0.5).astype(np.float32)
        
        correct_pixels = (pred_binary == gt_binary).sum()
        total_pixels = pred_binary.size
        
        return correct_pixels / total_pixels
    
    def compute_average_precision(self, pred_mask, gt_mask):
        """Compute Average Precision for pixel-level scores"""
        pred_flat = pred_mask.flatten()
        gt_flat = gt_mask.flatten()
        
        # Sort by prediction scores
        sorted_indices = np.argsort(pred_flat)[::-1]
        pred_sorted = pred_flat[sorted_indices]
        gt_sorted = gt_flat[sorted_indices]
        
        # Calculate precision and recall
        tp = np.cumsum(gt_sorted)
        fp = np.cumsum(1 - gt_sorted)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (gt_flat.sum() + 1e-8)
        
        # Calculate AP
        ap = np.trapz(precision, recall)
        return ap
    
    def compute_pixel_auc(self, pred_mask, gt_mask):
        """Compute AUC treating each pixel as a sample"""
        pred_flat = pred_mask.flatten()
        gt_flat = gt_mask.flatten()
        
        if len(np.unique(gt_flat)) < 2:
            return 0.5  # No positive samples
        
        auc = roc_auc_score(gt_flat, pred_flat)
        return auc
    
    def compute_correlation(self, pred_mask, gt_mask):
        """Compute Pearson correlation between soft mask intensities and GT mask values"""
        pred_flat = pred_mask.flatten()
        gt_flat = gt_mask.flatten()
        
        correlation, _ = pearsonr(pred_flat, gt_flat)
        return correlation if not np.isnan(correlation) else 0.0
    
    def compute_mask_metrics(self, pred_masks, gt_masks, threshold=0.5):
        """Compute all mask evaluation metrics"""
        if len(pred_masks.shape) == 4:
            # Batch of masks
            ious = []
            pixel_accuracies = []
            aps = []
            pixel_aucs = []
            correlations = []
            
            for pred, gt in zip(pred_masks, gt_masks):
                ious.append(self.compute_iou(pred, gt, threshold))
                pixel_accuracies.append(self.compute_pixel_accuracy(pred, gt, threshold))
                aps.append(self.compute_average_precision(pred, gt))
                pixel_aucs.append(self.compute_pixel_auc(pred, gt))
                correlations.append(self.compute_correlation(pred, gt))
            
            return {
                'iou': np.mean(ious),
                'pixel_accuracy': np.mean(pixel_accuracies),
                'average_precision': np.mean(aps),
                'pixel_auc': np.mean(pixel_aucs),
                'correlation': np.mean(correlations)
            }
        else:
            # Single mask
            return {
                'iou': self.compute_iou(pred_masks, gt_masks, threshold),
                'pixel_accuracy': self.compute_pixel_accuracy(pred_masks, gt_masks, threshold),
                'average_precision': self.compute_average_precision(pred_masks, gt_masks),
                'pixel_auc': self.compute_pixel_auc(pred_masks, gt_masks),
                'correlation': self.compute_correlation(pred_masks, gt_masks)
            }

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = self._find_target_layer(target_layer)
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _find_target_layer(self, layer_name):
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        raise ValueError(f"Layer {layer_name} not found in model.")

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def _save_activation(self, module, input, output):
        self.activations = output

    def _register_hooks(self):
        handle_act = self.target_layer.register_forward_hook(self._save_activation)
        handle_grad = self.target_layer.register_full_backward_hook(self._save_gradient)
        self.hook_handles.extend([handle_act, handle_grad])

    def __call__(self, x, class_idx=1):
        self.model.eval()
        cls_logits, _ = self.model(x)
        self.model.zero_grad()
        score = cls_logits[:, class_idx].sum()
        score.backward()

        if self.gradients is None or self.activations is None:
            self.remove_hooks()
            raise RuntimeError("Gradients or activations not captured. Check hook implementation.")

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(activations, dim=1).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
        return cv2.resize(heatmap, (x.shape[2], x.shape[3]))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

class MaskVisualizer:
    """Utility class for visualizing masks and heatmaps"""
    
    def __init__(self, output_dir='outputs'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_overlay(self, image, mask, title, filename):
        """Save image with mask overlay"""
        import matplotlib.pyplot as plt
        
        # Convert tensor to numpy if needed
        if torch.is_tensor(image):
            image = image.permute(1, 2, 0).cpu().numpy()
        if torch.is_tensor(mask):
            mask = mask.squeeze().cpu().numpy()
        
        # Normalize image
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Create overlay
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.imshow(mask, alpha=0.6, cmap='hot')
        plt.title(f'{title} Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_comparison(self, image, masks_dict, filename):
        """Save comparison of multiple masks"""
        import matplotlib.pyplot as plt
        
        # Convert tensor to numpy if needed
        if torch.is_tensor(image):
            image = image.permute(1, 2, 0).cpu().numpy()
        
        # Normalize image
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        n_masks = len(masks_dict)
        fig, axes = plt.subplots(2, n_masks + 1, figsize=(5 * (n_masks + 1), 10))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(image)
        axes[1, 0].set_title('Original')
        axes[1, 0].axis('off')
        
        # Each mask
        for i, (name, mask) in enumerate(masks_dict.items()):
            if torch.is_tensor(mask):
                mask = mask.squeeze().cpu().numpy()
            
            # Soft mask
            axes[0, i + 1].imshow(mask, cmap='hot')
            axes[0, i + 1].set_title(f'{name} (Soft)')
            axes[0, i + 1].axis('off')
            
            # Binary mask
            binary_mask = (mask > 0.5).astype(np.float32)
            axes[1, i + 1].imshow(binary_mask, cmap='hot')
            axes[1, i + 1].set_title(f'{name} (Binary)')
            axes[1, i + 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
