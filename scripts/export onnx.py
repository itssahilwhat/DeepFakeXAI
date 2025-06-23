import torch
from src.config import Config
from src.model import EfficientNetLiteTemporal


def export_to_onnx(checkpoint_path, output_path):
    model = EfficientNetLiteTemporal(num_classes=1, pretrained=False).to(Config.DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 3, *Config.INPUT_SIZE).to(Config.DEVICE)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"✅ ONNX model exported to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--output", type=str, default="onnx/model.onnx", help="Output path for ONNX model")
    args = parser.parse_args()
    export_to_onnx(args.checkpoint, args.output)