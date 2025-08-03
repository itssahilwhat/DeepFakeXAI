#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append('.')

from src.core.config import Config
from src.deployment import MobileExporter
from src.preprocessing.dataset import get_dataloader

def main():
    """Unified export script for mobile deployment"""
    parser = argparse.ArgumentParser(description='Export models for mobile deployment')
    parser.add_argument('--models', type=str, nargs='+', 
                       choices=['gradcam', 'patch', 'attention', 'mobilenetv3'],
                       default=['gradcam', 'patch', 'attention', 'mobilenetv3'],
                       help='Models to export')
    parser.add_argument('--calibration', action='store_true',
                       help='Use calibration data for static quantization')
    
    args = parser.parse_args()
    
    print("🚀 Starting Mobile Model Export")
    print("="*50)
    
    # Initialize config
    config = Config()
    
    # Create exporter
    exporter = MobileExporter(config)
    
    # Get calibration data if requested
    calibration_loader = None
    if args.calibration:
        print("Loading calibration data...")
        try:
            calibration_loader = get_dataloader('valid', config, batch_size_override=32)
            print(f"Calibration data loaded: {len(calibration_loader)} batches")
        except Exception as e:
            print(f"Warning: Could not load calibration data: {e}")
    
    # Export specified models
    print("\nStarting model export...")
    results = exporter.export_all_models(
        checkpoint_dir=config.CHECKPOINT_DIR,
        calibration_loader=calibration_loader
    )
    
    # Print detailed results
    print("\n" + "="*60)
    print("DETAILED EXPORT RESULTS")
    print("="*60)
    
    for model_type, result in results.items():
        if model_type in args.models and 'error' not in result:
            print(f"\n📊 {model_type.upper()} MODEL:")
            print(f"   Checkpoint: {result['checkpoint_path']}")
            
            print(f"\n   📁 Exports:")
            for export_type, export_path in result['exports'].items():
                print(f"      {export_type}: {export_path}")
            
            print(f"\n   ⚡ Benchmarks:")
            for benchmark_type, benchmark in result['benchmarks'].items():
                print(f"      {benchmark_type}:")
                print(f"        - Inference: {benchmark['avg_inference_time_ms']:.2f}ms ± {benchmark['std_inference_time_ms']:.2f}ms")
                print(f"        - FPS: {benchmark['fps']:.1f}")
                print(f"        - Size: {benchmark['model_size_mb']:.2f}MB")
        elif model_type in args.models:
            print(f"\n❌ {model_type.upper()}: FAILED")
            print(f"   Error: {result['error']}")
    
    # Check performance targets
    print("\n" + "="*60)
    print("PERFORMANCE TARGET CHECK")
    print("="*60)
    
    target_size_mb = config.TARGET_MODEL_SIZE_MB
    target_time_ms = config.TARGET_INFERENCE_TIME_MS
    
    for model_type, result in results.items():
        if model_type in args.models and 'error' not in result:
            print(f"\n🎯 {model_type.upper()}:")
            
            # Check ONNX performance
            if 'onnx' in result['benchmarks']:
                onnx_bench = result['benchmarks']['onnx']
                size_ok = onnx_bench['model_size_mb'] <= target_size_mb
                time_ok = onnx_bench['avg_inference_time_ms'] <= target_time_ms
                
                print(f"   ONNX Model:")
                print(f"     Size: {onnx_bench['model_size_mb']:.2f}MB / {target_size_mb}MB {'✅' if size_ok else '❌'}")
                print(f"     Time: {onnx_bench['avg_inference_time_ms']:.2f}ms / {target_time_ms}ms {'✅' if time_ok else '❌'}")
                print(f"     FPS: {onnx_bench['fps']:.1f}")
            
            # Check quantized performance
            if 'quantized' in result['benchmarks']:
                quant_bench = result['benchmarks']['quantized']
                size_ok = quant_bench['model_size_mb'] <= target_size_mb
                time_ok = quant_bench['avg_inference_time_ms'] <= target_time_ms
                
                print(f"   Quantized Model:")
                print(f"     Size: {quant_bench['model_size_mb']:.2f}MB / {target_size_mb}MB {'✅' if size_ok else '❌'}")
                print(f"     Time: {quant_bench['avg_inference_time_ms']:.2f}ms / {target_time_ms}ms {'✅' if time_ok else '❌'}")
                print(f"     FPS: {quant_bench['fps']:.1f}")
    
    print(f"\n📁 All exported models saved to: {exporter.export_dir}")
    print("✅ Mobile export completed!")

if __name__ == "__main__":
    main() 