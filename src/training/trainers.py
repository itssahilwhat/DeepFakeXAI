#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from .base_trainer import BaseTrainer
from src.detection.deepfake_models import GradCAMClassifier, PatchForensicsModel, MobileNetV3Classifier, UNet, EfficientNetClassifier, ResNetSE, AttentionModel, DeepfakeDetectionModel

class GradCAMTrainer(BaseTrainer):
    """Trainer for GradCAM-based models"""
    def __init__(self, config):
        super().__init__(config, model_name="gradcam")
    
    def _create_model(self):
        return GradCAMClassifier()

class PatchForensicsTrainer(BaseTrainer):
    """Trainer for Patch-Forensics models"""
    def __init__(self, config):
        super().__init__(config, model_name="patch")
    
    def _create_model(self):
        return PatchForensicsModel()
    
    def _forward_pass(self, batch):
        """Forward pass for patch-based models"""
        images, masks, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        logits, patch_logits = self.model(images)
        return logits, labels, masks
    
    def _validation_forward(self, batch):
        """Validation forward pass for patch-based models"""
        images, masks, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        with torch.no_grad():
            logits, patch_logits = self.model(images)
        return logits, labels, masks

class MobileNetV3Trainer(BaseTrainer):
    """Trainer for MobileNetV3 models"""
    def __init__(self, config):
        super().__init__(config, model_name="mobilenetv3")
    
    def _create_model(self):
        return MobileNetV3Classifier()

class UNetTrainer(BaseTrainer):
    """Trainer for U-Net segmentation model"""
    def __init__(self, config):
        super().__init__(config, model_name="unet")
    
    def _create_model(self):
        return UNet(num_classes=1)
    
    def _forward_pass(self, batch):
        """Forward pass for U-Net"""
        images, masks, labels = batch
        images = images.to(self.device)
        masks = masks.to(self.device) if masks is not None else None
        
        segmentation_mask = self.model(images)
        return segmentation_mask, masks, labels
    
    def _validation_forward(self, batch):
        """Validation forward pass for U-Net"""
        images, masks, labels = batch
        images = images.to(self.device)
        masks = masks.to(self.device) if masks is not None else None
        
        with torch.no_grad():
            segmentation_mask = self.model(images)
        return segmentation_mask, masks, labels
    
    def _calculate_loss(self, outputs, targets, masks=None):
        """Calculate segmentation loss"""
        segmentation_mask, target_masks = outputs, targets
        
        if target_masks is not None:
            return F.binary_cross_entropy(segmentation_mask, target_masks)
        return F.binary_cross_entropy(segmentation_mask, torch.zeros_like(segmentation_mask))

class EfficientNetTrainer(BaseTrainer):
    """Trainer for EfficientNet models"""
    def __init__(self, config, backbone='efficientnet_b0'):
        self.backbone = backbone
        super().__init__(config, model_name=f"efficientnet_{backbone}")
    
    def _create_model(self):
        return EfficientNetClassifier(backbone=self.backbone)

class ResNetSETrainer(BaseTrainer):
    """Trainer for ResNet-SE models"""
    def __init__(self, config, backbone='resnet34'):
        self.backbone = backbone
        super().__init__(config, model_name=f"resnetse_{backbone}")
    
    def _create_model(self):
        return ResNetSE(backbone=self.backbone)

class AttentionTrainer(BaseTrainer):
    """Trainer for attention-based models"""
    def __init__(self, config):
        super().__init__(config, model_name="attention")
    
    def _create_model(self):
        return AttentionModel()
    
    def _forward_pass(self, batch):
        """Forward pass with attention"""
        images, masks, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        logits, attention_map = self.model(images)
        return logits, labels, masks, attention_map
    
    def _validation_forward(self, batch):
        """Validation forward pass with attention"""
        images, masks, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        with torch.no_grad():
            logits, attention_map = self.model(images)
        return logits, labels, masks, attention_map

class MultiTaskTrainer(BaseTrainer):
    """Trainer for multi-task detection and segmentation"""
    def __init__(self, config, backbone='efficientnet_b0'):
        self.backbone = backbone
        super().__init__(config, model_name=f"multitask_{backbone}")
    
    def _create_model(self):
        return DeepfakeDetectionModel(backbone=self.backbone)
    
    def _forward_pass(self, batch):
        """Forward pass for multi-task model"""
        images, masks, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        masks = masks.to(self.device) if masks is not None else None
        
        detection_logits, segmentation_mask = self.model(images)
        return detection_logits, segmentation_mask, labels, masks
    
    def _validation_forward(self, batch):
        """Validation forward pass for multi-task model"""
        images, masks, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        masks = masks.to(self.device) if masks is not None else None
        
        with torch.no_grad():
            detection_logits, segmentation_mask = self.model(images)
        return detection_logits, segmentation_mask, labels, masks
    
    def _calculate_loss(self, outputs, targets, masks=None):
        """Calculate multi-task loss"""
        detection_logits, segmentation_mask = outputs
        labels, target_masks = targets
        
        # Detection loss
        detection_loss = F.cross_entropy(detection_logits, labels)
        
        # Segmentation loss (if masks available)
        segmentation_loss = 0
        if target_masks is not None:
            segmentation_loss = F.binary_cross_entropy(segmentation_mask, target_masks)
        
        # Total loss with configurable weights
        detection_weight = getattr(self.config, 'DETECTION_WEIGHT', 1.0)
        segmentation_weight = getattr(self.config, 'SEGMENTATION_WEIGHT', 1.0)
        
        total_loss = (detection_weight * detection_loss + 
                     segmentation_weight * segmentation_loss)
        
        return total_loss 