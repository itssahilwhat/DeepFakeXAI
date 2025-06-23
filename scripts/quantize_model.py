import torch
from torch.quantization import QConfig, FakeQuantize
from torch.quantization.observer import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver
from src.config import Config
from src.model import EfficientNetLiteTemporal


def quantize_model(checkpoint_path, output_path):
    model = EfficientNetLiteTemporal(num_classes=1, pretrained=False).to(Config.DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    model.qconfig = QConfig(
        activation=FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            quant_min=0,
            quant_max=255,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
            reduce_range=False
        ),
        weight=FakeQuantize.with_args(
            observer=MovingAveragePerChannelMinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
            reduce_range=False
        )
    )
    torch.quantization.prepare(model, inplace=False)

    # Calibration with dummy data
    dummy_input = torch.randn(10, 3, *Config.INPUT_SIZE).to(Config.DEVICE)
    model(dummy_input)

    quantized_model = torch.quantization.convert(model, inplace=False)
    traced_model = torch.jit.trace(quantized_model, dummy_input[0:1])
    torch.jit.save(traced_model, output_path)
    print(f"✅ Quantized model saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--output", type=str, default="checkpoints/quantized_model.pth", help="Output path for quantized model")
    args = parser.parse_args()
    quantize_model(args.checkpoint, args.output)