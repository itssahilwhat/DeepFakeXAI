import torch
import torch.nn as nn
torch.backends.cudnn.benchmark = True
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.config import det_backbone

class DetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        self.net = timm.create_model(det_backbone, pretrained=True, num_classes=2)
    def forward(self, x):
        return self.net(x)

class SegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        # Use a simple encoder-decoder architecture
        self.encoder = timm.create_model(det_backbone, pretrained=True, features_only=True)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        features = self.encoder(x)
        # Use the last feature map
        x = features[-1]
        return self.decoder(x)