import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.1, reduction='mean', pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.pos_weight = pos_weight
        
    def forward(self, inputs, targets):
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        if inputs.dim() > 1 and inputs.size(1) > 1:
            inputs = inputs[:, 1] - inputs[:, 0]
        
        # Use pos_weight if provided for explicit class balancing
        if self.pos_weight is not None:
            bce_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, 
                pos_weight=self.pos_weight.to(inputs.device),
                reduction='none'
            )
        else:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05, disable_torch_grad_focal_loss=False):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        
    def forward(self, x, y):
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        los_pos = y * torch.log(xs_pos.clamp(min=self.clip))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.clip))

        loss = los_pos + los_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, square_denominator=False, with_logits=True):
        super().__init__()
        self.smooth = smooth
        self.square_denominator = square_denominator
        self.with_logits = with_logits

    def forward(self, input, target):
        if self.with_logits:
            input = torch.sigmoid(input)
        
        flat_input = input.view(-1)
        flat_target = target.view(-1)
        
        intersection = (flat_input * flat_target).sum()
        
        if self.square_denominator:
            denominator = (flat_input * flat_input).sum() + (flat_target * flat_target).sum()
        else:
            denominator = flat_input.sum() + flat_target.sum()
        
        loss = 1 - ((2 * intersection + self.smooth) / (denominator + self.smooth))
        
        return loss

class IoULoss(nn.Module):
    def __init__(self, smooth=1.0, with_logits=True):
        super().__init__()
        self.smooth = smooth
        self.with_logits = with_logits

    def forward(self, input, target):
        if self.with_logits:
            input = torch.sigmoid(input)
        
        flat_input = input.view(-1)
        flat_target = target.view(-1)
        
        intersection = (flat_input * flat_target).sum()
        union = flat_input.sum() + flat_target.sum() - intersection
        
        loss = 1 - ((intersection + self.smooth) / (union + self.smooth))
        
        return loss

class CombinedSegmentationLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=0.5, iou_weight=0.3):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.iou_loss = IoULoss()
        
    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        iou = self.iou_loss(pred, target)
        
        total_loss = (self.bce_weight * bce + 
                     self.dice_weight * dice + 
                     self.iou_weight * iou)
        
        return total_loss

class CombinedLoss(nn.Module):
    def __init__(self, cls_weight=1.0, seg_weight=0.0, focal_weight=1.0):
        super().__init__()
        self.cls_weight = cls_weight
        self.seg_weight = seg_weight
        self.focal_weight = focal_weight
        
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        
    def forward(self, cls_output, seg_output, cls_target, seg_target):
        if self.cls_weight > 0:
            cls_loss = self.focal_loss(cls_output, cls_target)
        else:
            cls_loss = torch.tensor(0.0, device=cls_output.device)
        
        if self.seg_weight > 0 and seg_output is not None:
            seg_loss = self.dice_loss(seg_output, seg_target)
        else:
            seg_loss = torch.tensor(0.0, device=cls_output.device)
        
        total_loss = (self.cls_weight * cls_loss + self.seg_weight * seg_loss)
        return total_loss, cls_loss, seg_loss

class AdvancedLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, margin=0.1, temperature=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.margin = margin
        self.temperature = temperature
        
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)
        self.center_loss = CenterLoss(margin=margin)
        
    def forward(self, logits, targets, features=None):
        focal_loss = self.focal_loss(logits, targets)
        
        contrastive_loss = 0.0
        if features is not None:
            contrastive_loss = self.contrastive_loss(features, targets)
        
        center_loss = 0.0
        if features is not None:
            center_loss = self.center_loss(features, targets)
        
        total_loss = focal_loss + 0.1 * contrastive_loss + 0.05 * center_loss
        return total_loss

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features, targets):
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        targets = targets.unsqueeze(1)
        positive_mask = (targets == targets.T).float()
        negative_mask = 1 - positive_mask
        positive_mask.fill_diagonal_(0)
        
        positive_loss = -torch.log(torch.exp(similarity_matrix) + 1e-8) * positive_mask
        negative_loss = torch.log(1 + torch.exp(similarity_matrix)) * negative_mask
        
        positive_loss = positive_loss.sum() / (positive_mask.sum() + 1e-8)
        negative_loss = negative_loss.sum() / (negative_mask.sum() + 1e-8)
        
        return positive_loss + negative_loss

class CenterLoss(nn.Module):
    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin
        
    def forward(self, features, targets):
        unique_targets = torch.unique(targets)
        centers = []
        
        for target in unique_targets:
            mask = (targets == target)
            if mask.sum() > 0:
                center = features[mask].mean(dim=0)
                centers.append(center)
        
        if len(centers) < 2:
            return torch.tensor(0.0, device=features.device)
        
        centers = torch.stack(centers)
        distances = torch.cdist(features, centers)
        min_distances, _ = distances.min(dim=1)
        center_loss = min_distances.mean()
        
        return center_loss

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = F.log_softmax(pred, dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data, self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        logprobs = F.log_softmax(pred, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
