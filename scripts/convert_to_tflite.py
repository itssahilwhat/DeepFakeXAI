import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config import Config

try:
    from onnx_tf.backend import prepare
    import tensorflow as tf
except ImportError:
    print('Please install onnx-tf and tensorflow: pip install onnx-tf tensorflow')
    exit(1)

def main():
    onnx_path = os.path.join(Config.ONNX_DIR, f'deepfake_{Config.BACKBONE}.onnx')
    tflite_path = os.path.join(Config.ONNX_DIR, f'deepfake_{Config.BACKBONE}.tflite')
    if not os.path.exists(onnx_path):
        print(f'ONNX model not found at {onnx_path}')
        return
    print(f'Converting {onnx_path} to TensorFlow Lite...')
    tf_rep = prepare(onnx_path)
    tf_model_path = os.path.join(Config.ONNX_DIR, f'deepfake_{Config.BACKBONE}_tf')
    tf_rep.export_graph(tf_model_path)
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f'TensorFlow Lite model saved to {tflite_path}')

if __name__ == "__main__":
    main() 