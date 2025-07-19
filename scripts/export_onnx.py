import os
import torch
from src.config import Config
from src.model import MultiTaskDeepfakeModel

def main():
    os.makedirs(Config.ONNX_DIR, exist_ok=True)
    model = MultiTaskDeepfakeModel(
        backbone_name=Config.BACKBONE,
        num_classes=Config.NUM_CLASSES,
        pretrained=False,
        dropout=Config.DROPOUT,
        segmentation=Config.SEGMENTATION,
        attention=Config.ATTENTION
    )
    model.load_state_dict(torch.load(os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth'), map_location='cpu'))
    model.eval()
    onnx_path = os.path.join(Config.ONNX_DIR, f'deepfake_{Config.BACKBONE}.onnx')
    quant_onnx_path = os.path.join(Config.ONNX_DIR, f'deepfake_{Config.BACKBONE}_quant.onnx')
    dummy_input = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=['class_output', 'segmentation_output'],
            opset_version=12,
            do_constant_folding=True,
            dynamic_axes={'input': {0: 'batch_size'}, 'class_output': {0: 'batch_size'}, 'segmentation_output': {0: 'batch_size'}}
        )
    print(f'Standard ONNX model exported to {onnx_path}')
    try:
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
        )
        quantized_model.eval()
        with torch.no_grad():
            torch.onnx.export(
                quantized_model,
                dummy_input,
                quant_onnx_path,
                input_names=['input'],
                output_names=['class_output', 'segmentation_output'],
                opset_version=12,
                do_constant_folding=True,
                dynamic_axes={'input': {0: 'batch_size'}, 'class_output': {0: 'batch_size'}, 'segmentation_output': {0: 'batch_size'}}
            )
        print(f'Quantized ONNX model exported to {quant_onnx_path}')
    except Exception as e:
        print(f'Quantized export failed: {e}')

if __name__ == "__main__":
    main() 