import torch
import torch.onnx
import os
import logging
from src.config import Config
from src.model import EfficientNetLiteTemporal

def export_to_onnx(model_path, output_path=None):
    """Export trained model to ONNX format for deployment"""
    logging.info("🔄 Starting ONNX export...")
    
    if output_path is None:
        output_path = os.path.join(Config.ONNX_DIR, "deepfake_detector.onnx")
    
    os.makedirs(Config.ONNX_DIR, exist_ok=True)
    
    # Load model
    model = EfficientNetLiteTemporal(pretrained=False).to(Config.DEVICE)
    checkpoint = torch.load(model_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, *Config.INPUT_SIZE).to(Config.DEVICE)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['classification', 'segmentation'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'classification': {0: 'batch_size'},
            'segmentation': {0: 'batch_size'}
        }
    )
    
    logging.info(f"✅ ONNX model exported to {output_path}")
    return output_path

if __name__ == "__main__":
    Config.setup_logging()
    model_path = os.path.join(Config.CHECKPOINT_DIR, "best_combined.pth")
    export_to_onnx(model_path)