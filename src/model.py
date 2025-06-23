import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.checkpoint import checkpoint


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=False)  # Avoid inplace for stability

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
        self.encoder_channels = encoder_channels[::-1]  # [1280, 224, 80, 40, 24]
        self.up4 = nn.ConvTranspose2d(self.encoder_channels[0], 128, 2, 2)  # Cap at 128
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
            out_ch = min(x.shape[1], 128)  # Cap at 128
            setattr(self, name, ConvBlock(in_ch, out_ch).to(x.device))
        return getattr(self, name)(cat)

    def _resize(self, x, target):
        return F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=False)


class EfficientNetLiteTemporal(nn.Module):
    def __init__(self, num_classes=1, pretrained=True, dropout_rate=0.2):
        super(EfficientNetLiteTemporal, self).__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        self.stem = nn.Sequential(backbone.features[0])
        self.blocks = nn.Sequential(*backbone.features[1:])
        self.stage_idxs = [2, 4, 6, 8]
        self.out_indices = set(self.stage_idxs)
        self.channels = [24, 40, 80, 224, 1280]
        self.encoder = backbone.features
        self.decoder = UNetLiteDecoder(self.channels)
        self.out_conv = nn.Conv2d(24, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)

    def forward(self, x, prev_x=None, mc_dropout=False):
        if prev_x is not None:
            prev_x_np = prev_x.cpu().numpy().transpose(0, 2, 3, 1) * 255
            x_np = x.cpu().numpy().transpose(0, 2, 3, 1) * 255
            flow_batch = []
            for p, c in zip(prev_x_np, x_np):
                p_gray = cv2.cvtColor(p.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                c_gray = cv2.cvtColor(c.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                flow = self.flow.calc(p_gray, c_gray, None)
                flow_batch.append(torch.from_numpy(flow).float().permute(2, 0, 1))
            flow = torch.stack(flow_batch).to(x.device)
            x = torch.cat([x, flow], dim=1)
            if self.stem[0].in_channels != x.shape[1]:
                self.stem[0] = nn.Conv2d(x.shape[1], self.stem[0].out_channels, kernel_size=3, stride=2, padding=1, bias=False).to(x.device)

        skips = []
        for idx, layer in enumerate(self.encoder):
            x = checkpoint(layer, x, use_reentrant=False) if idx in self.out_indices else layer(x)
            if idx in self.out_indices:
                skips.append(x)
        bottleneck = x
        x = self.decoder(bottleneck, skips)
        if mc_dropout:
            x = self.dropout(x)  # Enable dropout during inference
        else:
            x = self.dropout(x) if self.training else x
        x = self.out_conv(x)
        return torch.sigmoid(x)


if __name__ == "__main__":
    model = EfficientNetLiteTemporal(pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    out = model(dummy_input)
    print("Model output shape:", out.shape)