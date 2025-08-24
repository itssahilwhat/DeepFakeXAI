import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ============================================================================
# DEEPFAKE DETECTION MODEL
# ============================================================================

class VanillaCNN(nn.Module):
    """Simple CNN for deepfake detection"""
    def __init__(self, num_conv=5):
        super().__init__()
        
        # Build encoder layers
        layers = []
        in_channels = 3
        out_channels = [32, 64, 128, 256, 512][:num_conv]
        
        for out_ch in out_channels:
            layers += [
                nn.Conv2d(in_channels, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ]
            in_channels = out_ch
        
        self.encoder = nn.Sequential(*layers)
        
        # Build classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),        # Global average pooling
            nn.Flatten(),                    # Flatten to 1D
            nn.Linear(in_channels, 128),     # First FC layer
            nn.ReLU(),
            nn.Dropout(0.4),                 # Prevent overfitting
            nn.Linear(128, 64),              # Second FC layer
            nn.ReLU(),
            nn.Dropout(0.3),                 # Additional dropout
            nn.Linear(64, 2)                 # Output: real/fake
        )

    def forward(self, x):
        x = self.encoder(x)                  # Extract features
        return self.classifier(x)            # Classify

# ============================================================================
# ATTENTION MECHANISMS
# ============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Two FC layers with reduction
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        
        # Squeeze: global average pooling
        y = self.avg_pool(x).view(batch_size, channels)
        
        # Excitation: learn channel weights
        y = self.fc(y).view(batch_size, channels, 1, 1)
        
        # Scale: apply attention weights
        return x * y.expand_as(x)

# ============================================================================
# SEGMENTATION BUILDING BLOCKS
# ============================================================================

class DoubleConv(nn.Module):
    """Double convolution block with optional SE attention"""
    def __init__(self, in_channels, out_channels, use_se=True):
        super().__init__()
        self.use_se = use_se
        
        # Two consecutive convolutions
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Optional SE attention
        if use_se:
            self.se = SEBlock(out_channels)
    
    def forward(self, x): 
        # First convolution
        x = self.relu(self.bn1(self.conv1(x)))
        
        # Second convolution
        x = self.relu(self.bn2(self.conv2(x)))
        
        # Apply SE attention if enabled
        if self.use_se:
            x = self.se(x)
        
        return x

# ============================================================================
# ENHANCED UNET FOR SEGMENTATION
# ============================================================================

class UNet(nn.Module):
    """Enhanced UNet with SE blocks for deepfake segmentation"""
    def __init__(self, base_channels=64, use_se=True):
        super().__init__()
        self.use_se = use_se
        
        # Encoder path (downsampling)
        self.down1 = DoubleConv(3, base_channels, use_se=use_se)
        self.down2 = DoubleConv(base_channels, base_channels*2, use_se=use_se)
        self.down3 = DoubleConv(base_channels*2, base_channels*4, use_se=use_se)
        self.down4 = DoubleConv(base_channels*4, base_channels*8, use_se=use_se)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck (lowest resolution)
        self.bottleneck = DoubleConv(base_channels*8, base_channels*16, use_se=use_se)

        # Decoder path (upsampling)
        self.up4 = nn.ConvTranspose2d(base_channels*16, base_channels*8, 2, stride=2)
        self.conv4 = DoubleConv(base_channels*16, base_channels*8, use_se=use_se)
        
        self.up3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, stride=2)
        self.conv3 = DoubleConv(base_channels*8, base_channels*4, use_se=use_se)
        
        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)
        self.conv2 = DoubleConv(base_channels*4, base_channels*2, use_se=use_se)
        
        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)
        self.conv1 = DoubleConv(base_channels*2, base_channels, use_se=use_se)

        # Output heads for different scales
        self.outc = nn.Conv2d(base_channels, 1, 1)           # Main output
        self.outc_2x = nn.Conv2d(base_channels*2, 1, 1)      # 2x downsampled
        self.outc_4x = nn.Conv2d(base_channels*4, 1, 1)      # 4x downsampled

    def forward(self, x):
        # Encoder path
        d1 = self.down1(x)                    # 224x224 -> 112x112
        p1 = self.pool(d1)                    # 112x112 -> 56x56
        
        d2 = self.down2(p1)                   # 56x56 -> 28x28
        p2 = self.pool(d2)                    # 28x28 -> 14x14
        
        d3 = self.down3(p2)                   # 14x14 -> 7x7
        p3 = self.pool(d3)                    # 7x7 -> 3x3
        
        d4 = self.down4(p3)                   # 3x3
        p4 = self.pool(d4)                    # 3x3 -> 1x1
        
        # Bottleneck
        bottleneck = self.bottleneck(p4)
        
        # Decoder path with skip connections
        # Up 4
        u4 = self.up4(bottleneck)             # 1x1 -> 3x3
        c4 = torch.cat([u4, d4], dim=1)      # Skip connection
        u4 = self.conv4(c4)                   # Process concatenated features
        
        # Up 3
        u3 = self.up3(u4)                     # 3x3 -> 7x7
        c3 = torch.cat([u3, d3], dim=1)      # Skip connection
        u3 = self.conv3(c3)                   # Process concatenated features
        
        # Up 2
        u2 = self.up2(u3)                     # 7x7 -> 14x14
        c2 = torch.cat([u2, d2], dim=1)      # Skip connection
        u2 = self.conv2(c2)                   # Process concatenated features
        
        # Up 1
        u1 = self.up1(u2)                     # 14x14 -> 28x28
        c1 = torch.cat([u1, d1], dim=1)      # Skip connection
        u1 = self.conv1(c1)                   # Process concatenated features
        
        # Multi-scale outputs (training only)
        if self.training:
            out_main = self.outc(u1)           # Main output: 28x28
            out_2x = self.outc_2x(u2)         # 2x output: 14x14
            out_4x = self.outc_4x(u3)         # 4x output: 7x7
            return out_main, out_2x, out_4x
        else:
            # Only main output for inference
            return self.outc(u1)

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'VanillaCNN',           # Simple CNN for detection
    'UNet',                 # Enhanced UNet for segmentation  
    'DoubleConv',           # Building block for UNet
    'SEBlock'               # Squeeze-and-Excitation attention
]