import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config import Config

try:
    from onnx_coreml import convert
except ImportError:
    print('Please install onnx-coreml: pip install onnx-coreml')
    exit(1)

def main():
    onnx_path = os.path.join(Config.ONNX_DIR, f'deepfake_{Config.BACKBONE}.onnx')
    coreml_path = os.path.join(Config.ONNX_DIR, f'deepfake_{Config.BACKBONE}.mlmodel')
    if not os.path.exists(onnx_path):
        print(f'ONNX model not found at {onnx_path}')
        return
    print(f'Converting {onnx_path} to CoreML...')
    mlmodel = convert(model=onnx_path)
    mlmodel.save(coreml_path)
    print(f'CoreML model saved to {coreml_path}')

if __name__ == "__main__":
    main() 