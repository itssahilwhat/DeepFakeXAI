import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from src.config import Config
import numpy as np
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn.functional as F


class LightweightConvBlock(nn.Module):
    """Ultra-lightweight convolution block for speed"""
    def __init__(self, in_ch, out_ch):
        super(LightweightConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True)  # ReLU6 for efficiency
        )

    def forward(self, x):
        return self.conv(x)


class FastDecoder(nn.Module):
    """Ultra-fast decoder for segmentation"""
    def __init__(self, in_channels=1280):  # Fixed: EfficientNet outputs 1280 channels
        super(FastDecoder, self).__init__()
        # Minimal decoder for speed
        self.up1 = nn.ConvTranspose2d(in_channels, 256, 2, 2)
        self.conv1 = LightweightConvBlock(256, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv2 = LightweightConvBlock(128, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv3 = LightweightConvBlock(64, 64)
        
        self.up4 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.conv4 = LightweightConvBlock(32, 32)
        
    def forward(self, x):
        x = self.up1(x)
        x = self.conv1(x)
        x = self.up2(x)
        x = self.conv2(x)
        x = self.up3(x)
        x = self.conv3(x)
        x = self.up4(x)
        x = self.conv4(x)
        return x


class EfficientNetLiteTemporal(nn.Module):
    def __init__(self, num_classes=1, pretrained=True, dropout_rate=None):
        super(EfficientNetLiteTemporal, self).__init__()
        
        dropout_rate = dropout_rate if dropout_rate is not None else Config.DROPOUT_RATE

        # Load EfficientNet-B0 backbone
        if pretrained:
            self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            self.backbone = efficientnet_b0(weights=None)
        
        # Remove the classifier
        self.backbone.classifier = nn.Identity()
        
        # XAI-compatible layer for GradCAM (minimal overhead)
        self.xai_layer = nn.Conv2d(1280, 1280, kernel_size=1, bias=False)  # Fixed: 1280 channels
        
        # Fast decoder for segmentation
        self.decoder = FastDecoder(1280)  # Fixed: 1280 channels
        
        # Final segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),  # Fixed: 32 input channels from decoder
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
            nn.Conv2d(16, num_classes, 1),
            nn.Sigmoid()
        )
        
        # Classification head for dual-head architecture
        if Config.USE_COLLABORATIVE:
            self.cls_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(dropout_rate),
                nn.Linear(1280, 256),  # Fixed: 1280 input channels
                nn.ReLU6(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
        
        # Minimal dropout
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x, prev_frames=None, mc_dropout=False):
        # Extract features from EfficientNet backbone efficiently
        x = self.backbone.features(x)

        # Apply XAI-compatible layer for GradCAM
        xai_output = self.xai_layer(x)

        # Apply minimal dropout
        x = self.dropout(x) if self.training or mc_dropout else x

        # Fast decoder
        x = self.decoder(x)

        # Segmentation output
        seg_output = self.seg_head(x)
        seg_output = F.interpolate(seg_output, size=Config.INPUT_SIZE, mode='bilinear', align_corners=False)

        # Classification output for dual-head architecture
        if Config.USE_COLLABORATIVE:
            # Use the final encoder features for classification
            cls_features = xai_output  # Use XAI layer output
            cls_output = self.cls_head(cls_features)
            return cls_output, seg_output
        else:
            return None, seg_output


if __name__ == "__main__":
    # Test the model
    model = EfficientNetLiteTemporal(pretrained=False, dropout_rate=0.1)
    dummy_input = torch.randn(2, 3, 224, 224)
    out = model(dummy_input)
    print("Model output shape:", out[1].shape if isinstance(out, tuple) else out.shape)
    print("Model architecture verified successfully")