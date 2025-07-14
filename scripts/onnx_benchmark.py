import time
import numpy as np
import onnxruntime as ort
import torch
import logging
from src.config import Config
import os

def benchmark_onnx(onnx_path, num_runs=100, batch_sizes=[1, 4, 8, 16]):
    """Benchmark ONNX model performance"""
    logging.info("⚡ Starting ONNX benchmark...")
    
    # Create ONNX runtime session
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    results = {}
    
    for batch_size in batch_sizes:
        logging.info(f"Testing batch size: {batch_size}")
        
        # Create dummy input
        dummy_input = np.random.randn(batch_size, 3, *Config.INPUT_SIZE).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            session.run(None, {'input': dummy_input})
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            session.run(None, {'input': dummy_input})
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        throughput = batch_size / avg_time
        
        results[batch_size] = {
            'avg_time': avg_time,
            'std_time': std_time,
            'throughput': throughput
        }
        
        logging.info(f"  Batch {batch_size}: {throughput:.1f} it/sec (±{std_time:.4f}s)")
    
    return results

def benchmark_pytorch_vs_onnx(model_path, onnx_path, num_runs=50):
    """Compare PyTorch vs ONNX performance"""
    logging.info("🔄 Comparing PyTorch vs ONNX performance...")
    
    # Load PyTorch model
    from src.model import EfficientNetLiteTemporal
    model = EfficientNetLiteTemporal(pretrained=False).to(Config.DEVICE)
    checkpoint = torch.load(model_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load ONNX model
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # Test data
    dummy_input = torch.randn(1, 3, *Config.INPUT_SIZE).to(Config.DEVICE)
    dummy_input_np = dummy_input.cpu().numpy()
    
    # PyTorch benchmark
    torch.cuda.synchronize()
    times_pytorch = []
    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            _ = model(dummy_input)
        torch.cuda.synchronize()
        end_time = time.time()
        times_pytorch.append(end_time - start_time)
    
    # ONNX benchmark
    times_onnx = []
    for _ in range(num_runs):
        start_time = time.time()
        _ = session.run(None, {'input': dummy_input_np})
        end_time = time.time()
        times_onnx.append(end_time - start_time)
    
    pytorch_avg = np.mean(times_pytorch)
    onnx_avg = np.mean(times_onnx)
    speedup = pytorch_avg / onnx_avg
    
    logging.info(f"PyTorch: {pytorch_avg:.4f}s ± {np.std(times_pytorch):.4f}s")
    logging.info(f"ONNX: {onnx_avg:.4f}s ± {np.std(times_onnx):.4f}s")
    logging.info(f"Speedup: {speedup:.2f}x")
    
    return {
        'pytorch_time': pytorch_avg,
        'onnx_time': onnx_avg,
        'speedup': speedup
    }

if __name__ == "__main__":
    Config.setup_logging()
    
    onnx_path = os.path.join(Config.ONNX_DIR, "deepfake_detector.onnx")
    model_path = os.path.join(Config.CHECKPOINT_DIR, "best_combined.pth")
    
    if os.path.exists(onnx_path):
        # Benchmark ONNX
        results = benchmark_onnx(onnx_path)
        
        # Compare with PyTorch
        if os.path.exists(model_path):
            comparison = benchmark_pytorch_vs_onnx(model_path, onnx_path)
    else:
        logging.error(f"ONNX model not found: {onnx_path}")
        logging.info("Run scripts/export_onnx.py first to export the model") 