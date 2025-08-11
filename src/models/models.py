import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.config import det_backbone

class VanillaCNN(nn.Module):
    """Vanilla CNN for deepfake detection as specified in requirements"""
    def __init__(self, num_conv=5):
        super().__init__()
        layers = []
        in_ch = 3
        out_chs = [32, 64, 128, 256, 512][:num_conv]
        for oc in out_chs:
            layers += [
                nn.Conv2d(in_ch, oc, kernel_size=3, padding=1),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ]
            in_ch = oc
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),        # N×C×1×1
            nn.Flatten(),                   # N×C
            nn.Linear(in_ch, 128),
            nn.ReLU(),
            nn.Dropout(0.4),                # Increased from 0.2 to 0.4
            nn.Linear(128, 64),             # Added intermediate layer
            nn.ReLU(),
            nn.Dropout(0.3),                # Additional dropout
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
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

class DoubleConv(nn.Module):
    """Double convolution block with SE attention for UNet"""
    def __init__(self, in_c, out_c, use_se=True):
        super().__init__()
        self.use_se = use_se
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        
        if use_se:
            self.se = SEBlock(out_c)
    
    def forward(self, x): 
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        if self.use_se:
            x = self.se(x)
        return x

class UNet(nn.Module):
    """Enhanced UNet with SE blocks and multi-scale supervision for deepfake segmentation"""
    def __init__(self, base_ch=64, use_se=True):
        super().__init__()
        self.use_se = use_se
        
        # Encoder with SE blocks
        self.down1 = DoubleConv(3, base_ch, use_se=use_se)
        self.down2 = DoubleConv(base_ch, base_ch*2, use_se=use_se)
        self.down3 = DoubleConv(base_ch*2, base_ch*4, use_se=use_se)
        self.down4 = DoubleConv(base_ch*4, base_ch*8, use_se=use_se)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_ch*8, base_ch*16, use_se=use_se)

        # Decoder with residual connections and SE blocks
        self.up4 = nn.ConvTranspose2d(base_ch*16, base_ch*8, 2, stride=2)
        self.conv4 = DoubleConv(base_ch*16, base_ch*8, use_se=use_se)
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.conv3 = DoubleConv(base_ch*8, base_ch*4, use_se=use_se)
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.conv2 = DoubleConv(base_ch*4, base_ch*2, use_se=use_se)
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.conv1 = DoubleConv(base_ch*2, base_ch, use_se=use_se)

        # Multi-scale output heads
        self.outc = nn.Conv2d(base_ch, 1, 1)
        self.outc_2x = nn.Conv2d(base_ch*2, 1, 1)  # 2x downsampled
        self.outc_4x = nn.Conv2d(base_ch*4, 1, 1)  # 4x downsampled

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        p1 = self.pool(d1)
        d2 = self.down2(p1)
        p2 = self.pool(d2)
        d3 = self.down3(p2)
        p3 = self.pool(d3)
        d4 = self.down4(p3)
        p4 = self.pool(d4)
        
        # Bottleneck
        bn = self.bottleneck(p4)
        
        # Decoder with skip connections
        u4 = self.up4(bn)
        c4 = torch.cat([u4, d4], dim=1)
        u4 = self.conv4(c4)
        
        u3 = self.up3(u4)
        c3 = torch.cat([u3, d3], dim=1)
        u3 = self.conv3(c3)
        
        u2 = self.up2(u3)
        c2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(c2)
        
        u1 = self.up1(u2)
        c1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv1(c1)
        
        # Multi-scale outputs (for training only)
        if self.training:
            # Main output
            out_main = self.outc(u1)
            # 2x downsampled output
            out_2x = self.outc_2x(u2)
            # 4x downsampled output  
            out_4x = self.outc_4x(u3)
            return out_main, out_2x, out_4x
        else:
            # Only return main output for inference
            return self.outc(u1)

class SwinAdapter(nn.Module):
    """Swin Transformer Adapter for feature refinement"""
    def __init__(self, in_channels, adapter_channels=256):
        super().__init__()
        self.in_channels = in_channels
        self.adapter_channels = adapter_channels
        
        # Multi-scale feature processing
        self.conv1 = nn.Conv2d(in_channels, adapter_channels//2, 1)
        self.conv3 = nn.Conv2d(in_channels, adapter_channels//4, 3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, adapter_channels//4, 5, padding=2)
        
        self.bn = nn.BatchNorm2d(adapter_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Multi-scale feature extraction
        f1 = self.conv1(x)
        f3 = self.conv3(x)
        f5 = self.conv5(x)
        
        # Concatenate and refine
        concat = torch.cat([f1, f3, f5], dim=1)
        return self.relu(self.bn(concat))

class DetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        self.net = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=2)
    def forward(self, x):
        return self.net(x)

class SegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        # Swin Transformer encoder
        self.encoder = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, features_only=True)
        
        # Get feature info for decoder input
        feature_info = self.encoder.feature_info.info
        in_channels = feature_info[-1]['num_chs']
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)
        )
    def forward(self, x):
        features = self.encoder(x)
        x = features[-1]
        return self.decoder(x)

class SwinHybridModel(nn.Module):
    """
    Swin Transformer-based Hybrid Deepfake Detection Model
    - Swin-T encoder with hierarchical attention
    - Multi-task: detection, segmentation, patch supervision
    - Reduced bias through better feature learning
    """
    def __init__(self, adapter_channels=256):
        super().__init__()
        import timm
        
        # Swin Transformer encoder
        self.encoder = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, features_only=True)
        
        # Get feature info
        feature_info = self.encoder.feature_info.info
        in_channels = feature_info[-1]['num_chs']
        
        # Swin adapter for feature refinement
        self.adapter = SwinAdapter(in_channels, adapter_channels)
        
        # Global classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(adapter_channels, 2)
        )
        
        # Segmentation decoder
        self.seg_decoder = nn.Sequential(
            nn.ConvTranspose2d(adapter_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)
        )
        
        # Patch supervision removed to fix bias
        # self.patch_head = nn.Conv2d(adapter_channels, 1, 1)
        
        # Unfreeze last stage for better adaptation
        self._unfreeze_last_stage()
            
    def _unfreeze_last_stage(self):
        """Unfreeze last Swin stage for better feature adaptation"""
        for name, param in self.encoder.named_parameters():
            if "layers.3" in name:  # Last stage
                param.requires_grad = True
            else:
                param.requires_grad = False
            
    def forward(self, x):
        # Get Swin features
        features = self.encoder(x)
        feature_map = features[-1]  # [B, H, W, C] format
        
        # Swin outputs [B, H, W, C], need to transpose to [B, C, H, W]
        feature_map = feature_map.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Apply adapter
        adapted_features = self.adapter(feature_map)
        
        # Global classification
        pooled = self.global_pool(adapted_features)
        logits_cls = self.classifier(pooled.view(pooled.size(0), -1))
        
        # Segmentation mask
        mask_logits = self.seg_decoder(adapted_features)
        
        # Patch supervision removed to fix bias
        # patch_logits = self.patch_head(adapted_features)
        
        return logits_cls, mask_logits
    
    def get_feature_map(self, x):
        """Get feature map for Grad-CAM"""
        features = self.encoder(x)
        feature_map = features[-1]  # [B, H, W, C] format
        
        # Swin outputs [B, H, W, C], need to transpose to [B, C, H, W]
        feature_map = feature_map.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        return self.adapter(feature_map)

# Alias for backward compatibility
HybridDeepfakeModel = SwinHybridModel

# Export all models for easy import
__all__ = ['VanillaCNN', 'UNet', 'DoubleConv', 'DetectionModel', 'SegmentationModel', 'SwinHybridModel', 'HybridDeepfakeModel']