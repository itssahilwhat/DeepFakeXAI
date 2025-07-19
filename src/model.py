import torch
import torch.nn as nn
import timm
from src.config import Config

class DecoderBlock(nn.Module):
    """
    A U-Net-style decoder block.
    It takes features from a lower level, upsamples them, and concatenates them
    with skip-connection features from the corresponding encoder level.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # Upsample the input feature map to double its height and width
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # A convolutional block to process the concatenated features
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_features):
        """
        Args:
            x (torch.Tensor): The input tensor from the previous (deeper) decoder block.
            skip_features (torch.Tensor): The skip-connection tensor from the corresponding encoder stage.
        """
        x = self.upsample(x)
        x = torch.cat([x, skip_features], dim=1)
        return self.conv(x)

class MultiTaskDeepfakeModel(nn.Module):
    """
    The final, definitive multi-task model with a U-Net decoder for segmentation.
    """
    def __init__(self, backbone_name=Config.BACKBONE, num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED, dropout=Config.DROPOUT, segmentation=Config.SEGMENTATION, **kwargs):
        super().__init__()
        self.segmentation = segmentation

        # 1. Encoder (Backbone)
        # We use timm to create the model with `features_only=True` to get intermediate
        # feature maps, which are essential for the decoder's skip connections.
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
        )

        # Get the number of output channels from each stage of the backbone
        encoder_channels = self.backbone.feature_info.channels()
        # Example for 'mobilenetv3_small_100': [16, 16, 24, 48, 576]

        # 2. Decoder (U-Net Architecture)
        # The decoder reconstructs the mask by progressively upsampling and refining features.
        if self.segmentation:
            # Unpack the channel counts for each encoder stage
            c0_in, c1_in, c2_in, c3_in, c4_in = encoder_channels

            # Decoder blocks are wired from the deepest features upwards
            self.decoder4 = DecoderBlock(c4_in, c3_in, 256)
            self.decoder3 = DecoderBlock(256, c2_in, 128)
            self.decoder2 = DecoderBlock(128, c1_in, 64)
            self.decoder1 = DecoderBlock(64, c0_in, 32)

            # Final upsampling and a 1x1 convolution to produce the single-channel mask
            self.seg_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.seg_head = nn.Conv2d(32, 1, kernel_size=1)

        # 3. Classification Head
        # This head operates on the deepest, most semantically rich feature map from the encoder.
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(encoder_channels[-1], num_classes)
        )

    def forward(self, x):
        # The backbone returns a list of feature maps, from shallowest to deepest
        encoder_features = self.backbone(x)

        # The classification head uses the final (deepest) feature map for the highest accuracy
        cls_logits = self.cls_head(encoder_features[-1])

        if not self.segmentation:
            return cls_logits, None

        # Unpack features for the decoder's skip connections
        c0, c1, c2, c3, c4 = encoder_features

        # Pass features up through the decoder, merging with skip connections at each stage
        d4 = self.decoder4(c4, c3)      # Start with the deepest features
        d3 = self.decoder3(d4, c2)
        d2 = self.decoder2(d3, c1)
        d1 = self.decoder1(d2, c0)      # End with the shallowest features

        # Produce the final, high-resolution segmentation map
        seg_logits = self.seg_head(self.seg_upsample(d1))

        return cls_logits, seg_logits