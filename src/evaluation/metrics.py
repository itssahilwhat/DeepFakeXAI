#!/usr/bin/env python3

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score, 
    confusion_matrix, average_precision_score
)

class ClassificationMetrics:
    """Classification metrics calculator"""
    
    @staticmethod
    def calculate_metrics(predictions, labels, probabilities):
        """Calculate all classification metrics"""
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        auc = roc_auc_score(labels, probabilities)
        ap = average_precision_score(labels, probabilities)
        cm = confusion_matrix(labels, predictions)
        
        # Per-class metrics
        class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
            labels, predictions, average=None
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'ap': ap,
            'confusion_matrix': cm.tolist(),
            'class_precision': class_precision.tolist(),
            'class_recall': class_recall.tolist(),
            'class_f1': class_f1.tolist(),
            'predictions': predictions,
            'labels': labels,
            'probabilities': probabilities
        }

class LocalizationMetrics:
    """Localization metrics calculator"""
    
    @staticmethod
    def calculate_iou(pred_mask, gt_mask):
        """Calculate Intersection over Union"""
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        return intersection / (union + 1e-8)
    
    @staticmethod
    def calculate_dice(pred_mask, gt_mask):
        """Calculate Dice coefficient"""
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        return (2 * intersection) / (pred_mask.sum() + gt_mask.sum() + 1e-8)
    
    @staticmethod
    def calculate_pbca(pred_mask, gt_mask):
        """Calculate Partial Boundary Correctness"""
        # Simplified PBCA implementation
        # In practice, this would involve boundary detection and matching
        return np.mean(pred_mask == gt_mask)
    
    @staticmethod
    def calculate_metrics(pred_masks, gt_masks):
        """Calculate all localization metrics"""
        ious = []
        dice_scores = []
        pbca_scores = []
        
        for pred_mask, gt_mask in zip(pred_masks, gt_masks):
            # Threshold predictions
            pred_mask = (pred_mask > 0.5).astype(np.float32)
            gt_mask = (gt_mask > 0.5).astype(np.float32)
            
            # Calculate metrics
            iou = LocalizationMetrics.calculate_iou(pred_mask, gt_mask)
            dice = LocalizationMetrics.calculate_dice(pred_mask, gt_mask)
            pbca = LocalizationMetrics.calculate_pbca(pred_mask, gt_mask)
            
            ious.append(iou)
            dice_scores.append(dice)
            pbca_scores.append(pbca)
        
        return {
            'iou_mean': np.mean(ious),
            'iou_std': np.std(ious),
            'dice_mean': np.mean(dice_scores),
            'dice_std': np.std(dice_scores),
            'pbca_mean': np.mean(pbca_scores),
            'pbca_std': np.std(pbca_scores),
            'ious': ious,
            'dice_scores': dice_scores,
            'pbca_scores': pbca_scores
        } 