import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import Config


class DiceLoss(nn.Module):
    """Optimized Dice Loss for segmentation tasks"""

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target, weights=None):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = (pred + target).sum(dim=(1, 2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice.mean()
        if weights is not None:
            loss = loss * weights
        return loss.mean()


class FocalLoss(nn.Module):
    """Focal Loss for handling hard examples"""
    
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, weights=None):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if weights is not None:
            F_loss = F_loss * weights
            
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


class TemporalConsistencyLoss(nn.Module):
    """Temporal consistency loss for video frames"""
    
    def __init__(self, window_size=3):
        super(TemporalConsistencyLoss, self).__init__()
        self.window_size = window_size
        
    def forward(self, current_pred, prev_predictions):
        if len(prev_predictions) == 0:
            return torch.tensor(0.0, device=current_pred.device, requires_grad=True)
            
        loss = 0.0
        current_sigmoid = torch.sigmoid(current_pred)
        
        for prev_pred in prev_predictions:
            prev_sigmoid = torch.sigmoid(prev_pred)
            # L1 loss for temporal consistency
            temp_loss = F.l1_loss(current_sigmoid, prev_sigmoid)
            loss += temp_loss
            
        return loss / len(prev_predictions)


class BoundaryLoss(nn.Module):
    """Boundary-aware loss for better segmentation"""
    
    def __init__(self, kernel_size=5):
        super(BoundaryLoss, self).__init__()
        self.kernel_size = kernel_size
        
    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        
        # Create boundary kernel
        kernel = torch.ones(1, 1, self.kernel_size, self.kernel_size).to(pred.device)
        
        # Extract boundaries
        pred_boundary = F.conv2d(pred_sigmoid, kernel, padding=self.kernel_size//2)
        target_boundary = F.conv2d(target, kernel, padding=self.kernel_size//2)
        
        # Boundary loss - use with_logits version for autocast safety
        boundary_loss = F.binary_cross_entropy_with_logits(pred_boundary, target_boundary)
        
        return boundary_loss


class HybridLoss(nn.Module):
    """Advanced hybrid loss combining multiple loss functions"""

    def __init__(self, weights=None):
        super(HybridLoss, self).__init__()
        self.weights = weights or Config.LOSS_WEIGHTS
        
        # Initialize loss components
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.focal_loss = FocalLoss()
        self.temporal_loss = TemporalConsistencyLoss(Config.TEMPORAL_WINDOW_SIZE)
        self.boundary_loss = BoundaryLoss(Config.BOUNDARY_KERNEL_SIZE)

    def forward(self, pred, target, prev_predictions=None, weights=None):
        # Ensure inputs are valid
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # Calculate individual losses
        dice_loss = self.dice_loss(pred, target, weights)
        bce_loss = self.bce_loss(pred, target)
        focal_loss = self.focal_loss(pred, target, weights)
        
        # Temporal consistency loss
        temporal_loss = self.temporal_loss(pred, prev_predictions) if prev_predictions else torch.tensor(0.0, device=pred.device)
        
        # Boundary loss
        boundary_loss = self.boundary_loss(pred, target)
        
        # Combine losses with weights
        total_loss = (
            self.weights["dice_weight"] * dice_loss +
            self.weights["bce_weight"] * bce_loss +
            self.weights["focal_weight"] * focal_loss +
            self.weights["temporal_weight"] * temporal_loss +
            0.1 * boundary_loss  # Small weight for boundary loss
        )
        
        # Clamp for stability
        return torch.clamp(total_loss, min=0.0, max=10.0)


class IoULoss(nn.Module):
    """IoU Loss for better segmentation performance"""
    
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = (pred + target - pred * target).sum(dim=(1, 2, 3))
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou.mean()


class ComboLoss(nn.Module):
    """Combo Loss: Dice + BCE + IoU"""
    
    def __init__(self, weights=None):
        super(ComboLoss, self).__init__()
        self.weights = weights or {"dice": 0.5, "bce": 0.3, "iou": 0.2}
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.iou_loss = IoULoss()
        
    def forward(self, pred, target):
        dice_loss = self.dice_loss(pred, target)
        bce_loss = self.bce_loss(pred, target)
        iou_loss = self.iou_loss(pred, target)
        
        return (
            self.weights["dice"] * dice_loss +
            self.weights["bce"] * bce_loss +
            self.weights["iou"] * iou_loss
        )