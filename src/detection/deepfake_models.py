import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import models

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResNetSE(nn.Module):
    """ResNet with SE blocks"""
    def __init__(self, backbone='resnet50', num_classes=2):
        super(ResNetSE, self).__init__()
        if backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
            self.feature_dim = 512
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.se1 = SEBlock(64)
        self.se2 = SEBlock(128)
        self.se3 = SEBlock(256)
        self.se4 = SEBlock(512)
        
        self.backbone.fc = nn.Linear(self.feature_dim, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

class EfficientNetClassifier(nn.Module):
    """EfficientNet classifier"""
    def __init__(self, num_classes=2, backbone='efficientnet_b0'):
        super(EfficientNetClassifier, self).__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=num_classes)
        
    def forward(self, x):
        return self.backbone(x)

class GradCAMClassifier(nn.Module):
    """EfficientNet-based classifier"""
    def __init__(self, num_classes=2):
        super(GradCAMClassifier, self).__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
        
    def forward(self, x):
        return self.backbone(x)

class PatchForensicsModel(nn.Module):
    """Patch-based forensics model"""
    def __init__(self, num_classes=2, patch_size=32):
        super(PatchForensicsModel, self).__init__()
        self.patch_size = patch_size
        
        self.patch_classifier = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
        
        self.global_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        patches = self._extract_patches(x)
        
        patch_features = []
        patch_logits = []
        
        for patch in patches:
            patch_feat = self.patch_classifier(patch)
            patch_features.append(patch_feat)
            patch_logit = self.global_classifier(patch_feat)
            patch_logits.append(patch_logit)
        
        patch_features = torch.stack(patch_features, dim=1)
        patch_logits = torch.stack(patch_logits, dim=1)
        
        global_feat = torch.mean(patch_features, dim=1)
        global_logit = self.global_classifier(global_feat)
        
        return global_logit, patch_logits
    
    def _extract_patches(self, x):
        b, c, h, w = x.shape
        patches = []
        
        for i in range(0, h - self.patch_size + 1, self.patch_size // 2):
            for j in range(0, w - self.patch_size + 1, self.patch_size // 2):
                patch = x[:, :, i:i+self.patch_size, j:j+self.patch_size]
                patches.append(patch)
        
        return patches

class AttentionModel(nn.Module):
    """Attention-based model for localization"""
    def __init__(self, num_classes=2):
        super(AttentionModel, self).__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
        
        self.attention = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, 1, 1),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        features = self.backbone.forward_features(x)
        
        attention_map = self.attention(features)
        
        attended_features = features * attention_map
        
        pooled_features = F.adaptive_avg_pool2d(attended_features, 1).squeeze(-1).squeeze(-1)
        
        logits = self.classifier(pooled_features)
        
        return logits, attention_map

class UNet(nn.Module):
    """U-Net for segmentation"""
    def __init__(self, num_classes=1):
        super(UNet, self).__init__()
        
        self.enc1 = self._make_layer(3, 64)
        self.enc2 = self._make_layer(64, 128)
        self.enc3 = self._make_layer(128, 256)
        self.enc4 = self._make_layer(256, 512)
        
        self.dec4 = self._make_layer(512, 256)
        self.dec3 = self._make_layer(256, 128)
        self.dec2 = self._make_layer(128, 64)
        self.dec1 = nn.Conv2d(64, num_classes, 1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        dec4 = self.dec4(torch.cat([self.upsample(enc4), enc3], dim=1))
        dec3 = self.dec3(torch.cat([self.upsample(dec4), enc2], dim=1))
        dec2 = self.dec2(torch.cat([self.upsample(dec3), enc1], dim=1))
        dec1 = self.dec1(dec2)
        
        return torch.sigmoid(dec1)

class DeepfakeDetectionModel(nn.Module):
    """Multi-task model for detection and segmentation"""
    def __init__(self, num_classes=2, backbone='efficientnet_b0'):
        super(DeepfakeDetectionModel, self).__init__()
        
        if backbone == 'efficientnet_b0':
            self.backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
        elif backbone == 'resnet50':
            self.backbone = timm.create_model('resnet50', pretrained=True, num_classes=num_classes)
        elif backbone == 'resnet34se':
            self.backbone = ResNetSE(backbone='resnet34', num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.segmentation_head = UNet(num_classes=1)
        
    def forward(self, x):
        detection_logits = self.backbone(x)
        segmentation_mask = self.segmentation_head(x)
        
        return detection_logits, segmentation_mask

class MobileNetV3Classifier(nn.Module):
    """MobileNetV3 for mobile deployment"""
    def __init__(self, num_classes=2):
        super(MobileNetV3Classifier, self).__init__()
        self.backbone = timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=num_classes)
        
    def forward(self, x):
        return self.backbone(x)
