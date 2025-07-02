import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from config import Config
import numpy as np
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.checkpoint import checkpoint


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_ch, out_ch),
            DepthwiseSeparableConv(out_ch, out_ch),
        )

    def forward(self, x):
        return self.conv(x)


class UNetLiteDecoder(nn.Module):
    def __init__(self, encoder_channels):
        super(UNetLiteDecoder, self).__init__()
        self.encoder_channels = encoder_channels[::-1]
        self.up4 = nn.ConvTranspose2d(self.encoder_channels[0], 128, 2, 2)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.up1 = nn.ConvTranspose2d(32, 24, 2, 2)
        self.up0 = nn.ConvTranspose2d(24, 24, 2, 2)
        self.conv1 = None
        self.conv2 = None
        self.conv3 = None
        self.conv4 = None
        self.conv0 = None

    def forward(self, x, skips):
        # Ensure we have at least 4 skip connections
        while len(skips) < 4:
            skips.append(torch.zeros_like(x))

        s1, s2, s3, s4 = skips[-1], skips[-2], skips[-3], skips[-4]
        x = self.up4(x)
        s3 = self._resize(s3, x)
        x = self._conv_init('conv1', x, s3)
        x = self.up3(x)
        s2 = self._resize(s2, x)
        x = self._conv_init('conv2', x, s2)
        x = self.up2(x)
        s1 = self._resize(s1, x)
        x = self._conv_init('conv3', x, s1)
        x = self.up1(x)
        x = self._conv_init('conv4', x, None)
        x = self.up0(x)
        x = self._conv_init('conv0', x, None)
        return x

    def _conv_init(self, name, x, skip):
        if skip is not None:
            cat = torch.cat([x, skip], dim=1)
        else:
            cat = x
        if getattr(self, name) is None:
            in_ch = cat.shape[1]
            out_ch = min(x.shape[1], 128)
            setattr(self, name, ConvBlock(in_ch, out_ch).to(x.device))
        return getattr(self, name)(cat)

    def _resize(self, x, target):
        return F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=False)


class EfficientNetLiteTemporal(nn.Module):
    def __init__(self, num_classes=1, pretrained=True, dropout_rate=0.2):
        super(EfficientNetLiteTemporal, self).__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)

        # Store the original first conv layer separately
        self.original_conv = backbone.features[0][0]  # Get the Conv2d from Conv2dNormActivation
        self.stem = nn.Sequential(
            nn.Conv2d(
                5 if Config.TEMPORAL_WINDOW_SIZE > 0 else 3,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        # Initialize with pretrained weights if available
        if pretrained:
            with torch.no_grad():
                self.stem[0].weight[:, :3] = self.original_conv.weight[:, :3]

        self.blocks = nn.Sequential(*backbone.features[1:])
        self.stage_idxs = [2, 4, 6, 8]
        self.out_indices = set(self.stage_idxs)
        self.channels = [24, 40, 80, 224, 1280]
        self.encoder = backbone.features
        self.decoder = UNetLiteDecoder(self.channels)
        self.out_conv = nn.Conv2d(24, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_rate)

        # Optical flow parameters
        self.flow_type = 'farneback'  # or 'lucaskanade'
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.fb_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        if Config.USE_COLLABORATIVE:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(1280, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 1)
            )

    def calculate_flow(self, prev_frame, current_frame):
        """Calculate optical flow with automatic fallback"""
        try:
            prev_np = prev_frame.mul(255).byte().cpu().numpy().transpose(0, 2, 3, 1)
            curr_np = current_frame.mul(255).byte().cpu().numpy().transpose(0, 2, 3, 1)
            flow_batch = []

            for p, c in zip(prev_np, curr_np):
                p_gray = cv2.cvtColor(p, cv2.COLOR_RGB2GRAY) if p.shape[-1] == 3 else p.squeeze()
                c_gray = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY) if c.shape[-1] == 3 else c.squeeze()

                if p_gray.shape != c_gray.shape:
                    c_gray = cv2.resize(c_gray, (p_gray.shape[1], p_gray.shape[0]))

                # Use Farneback by default
                flow = cv2.calcOpticalFlowFarneback(p_gray, c_gray, None, **self.fb_params)
                flow_batch.append(torch.from_numpy(flow))

            flow_tensor = torch.stack(flow_batch).permute(0, 3, 1, 2)
            return flow_tensor.to(current_frame.device)

        except Exception as e:
            print(f"Optical flow failed: {str(e)}")
            return torch.zeros(
                prev_frame.size(0), 2, prev_frame.size(2), prev_frame.size(3),
                device=current_frame.device
            )

    def forward(self, x, prev_x=None, mc_dropout=False):
        # Optical flow processing
        if Config.TEMPORAL_WINDOW_SIZE > 0:
            if prev_x is not None:
                with torch.no_grad():
                    flow = self.calculate_flow(prev_x, x)
            else:
                # Create dummy zeros
                flow = torch.zeros(
                    x.size(0), 2, x.size(2), x.size(3), device=x.device
                )
            # Always concatenate 2 channels
            x = torch.cat([x, flow], dim=1)

        # Rest of the forward pass
        skips = []
        x = self.stem(x)
        for idx, layer in enumerate(self.blocks):
            x = layer(x)
            if idx in self.out_indices:
                skips.append(x)
        bottleneck = x

        x = self.decoder(bottleneck, skips)
        x = self.dropout(x) if self.training or mc_dropout else x
        seg_output = self.out_conv(x)

        if Config.USE_COLLABORATIVE:
            cls_output = self.classifier(bottleneck)
            return cls_output, seg_output
        return None, seg_output


if __name__ == "__main__":
    # Test the model with different input scenarios
    model = EfficientNetLiteTemporal(pretrained=False)

    # Test case 1: Normal input
    dummy_input = torch.randn(2, 3, 224, 224)
    out = model(dummy_input)
    print("Model output shape:", out[1].shape if Config.USE_COLLABORATIVE else out.shape)

    # Test case 2: With temporal input
    prev_frame = torch.randn(2, 3, 224, 224)
    out = model(dummy_input, prev_x=prev_frame)
    print("Temporal output shape:", out[1].shape if Config.USE_COLLABORATIVE else out.shape)

    print("Model architecture verified successfully")