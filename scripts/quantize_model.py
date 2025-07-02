# scripts/quantize_model.py
import torch
from torch.quantization import quantize_dynamic
from src.config import Config
from src.model import EfficientNetLiteTemporal


def quantize_model(checkpoint_path, output_path):
    # Load model
    model = EfficientNetLiteTemporal(num_classes=1, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Dynamic quantization
    quantized_model = quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d, nn.ConvTranspose2d},
        dtype=torch.qint8
    )

    # Save quantized model
    torch.save({
        "state_dict": quantized_model.state_dict(),
        "quantized": True
    }, output_path)
    print(f"✅ Quantized model saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--output", type=str, default="checkpoints/quantized_model.pth", help="Output path")
    args = parser.parse_args()
    quantize_model(args.checkpoint, args.output)