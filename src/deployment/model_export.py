#!/usr/bin/env python3

import os
import sys
import torch
import torch.nn as nn
import torch.quantization as quantization
import onnx
import onnxruntime as ort
import numpy as np
import time
from pathlib import Path
from typing import Dict, Tuple, Optional
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.config import Config
from src.detection.deepfake_models import GradCAMClassifier, PatchForensicsModel, AttentionModel, MobileNetV3Classifier

class MobileExporter:
    """Mobile export functionality for deepfake detection models"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = config.DEVICE
        self.export_dir = Path(config.OUTPUT_DIR) / 'mobile_models'
        self.export_dir.mkdir(exist_ok=True)
        
        # Export settings
        self.img_size = config.IMG_SIZE
        self.batch_size = 1  # Mobile inference typically uses batch size 1
        
    def load_model(self, model_type: str, checkpoint_path: str) -> nn.Module:
        """Load trained model from checkpoint"""
        if model_type == 'gradcam':
            model = GradCAMClassifier(backbone='efficientnet_b3', num_classes=2)
        elif model_type == 'patch':
            model = PatchForensicsModel(backbone='efficientnet_b3', patch_size=37)
        elif model_type == 'attention':
            model = AttentionModel(backbone='efficientnet_b3')
        elif model_type == 'mobilenetv3':
            model = MobileNetV3Classifier(num_classes=2)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def export_torchscript(self, model: nn.Module, model_name: str) -> str:
        """Export model to TorchScript format"""
        print(f"Exporting {model_name} to TorchScript...")
        
        # Create dummy input
        dummy_input = torch.randn(self.batch_size, 3, self.img_size, self.img_size).to(self.device)
        
        # Trace the model
        traced_model = torch.jit.trace(model, dummy_input)
        
        # Save traced model
        output_path = self.export_dir / f"{model_name}_traced.pt"
        torch.jit.save(traced_model, output_path)
        
        print(f"TorchScript model saved to: {output_path}")
        return str(output_path)
    
    def export_onnx(self, model: nn.Module, model_name: str) -> str:
        """Export model to ONNX format"""
        print(f"Exporting {model_name} to ONNX...")
        
        # Create dummy input
        dummy_input = torch.randn(self.batch_size, 3, self.img_size, self.img_size).to(self.device)
        
        # Export to ONNX
        output_path = self.export_dir / f"{model_name}.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        print(f"ONNX model saved to: {output_path}")
        return str(output_path)
    
    def quantize_dynamic(self, model: nn.Module, model_name: str) -> nn.Module:
        """Apply dynamic quantization to model"""
        print(f"Applying dynamic quantization to {model_name}...")
        
        # Apply dynamic quantization
        quantized_model = quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        
        return quantized_model
    
    def quantize_static(self, model: nn.Module, model_name: str, calibration_loader) -> nn.Module:
        """Apply static quantization to model (requires calibration data)"""
        print(f"Applying static quantization to {model_name}...")
        
        # Prepare model for quantization
        model.eval()
        model.qconfig = quantization.get_default_qconfig('fbgemm')
        
        # Prepare the model for static quantization
        quantization.prepare(model, inplace=True)
        
        # Calibrate with calibration data
        with torch.no_grad():
            for images, _, _ in calibration_loader:
                if len(images) > 0:
                    model(images.to(self.device))
                    break  # Use first batch for calibration
        
        # Convert to quantized model
        quantized_model = quantization.convert(model, inplace=False)
        
        return quantized_model
    
    def benchmark_model(self, model, model_name: str, num_runs: int = 100) -> Dict:
        """Benchmark model performance"""
        print(f"Benchmarking {model_name}...")
        
        # Create dummy input
        dummy_input = torch.randn(self.batch_size, 3, self.img_size, self.img_size).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(dummy_input)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1000 / avg_time  # Frames per second
        
        # Calculate model size
        if hasattr(model, 'state_dict'):
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            model_size_mb = (param_size + buffer_size) / (1024 * 1024)
        else:
            model_size_mb = 0
        
        results = {
            'model_name': model_name,
            'avg_inference_time_ms': avg_time,
            'std_inference_time_ms': std_time,
            'fps': fps,
            'model_size_mb': model_size_mb,
            'num_runs': num_runs
        }
        
        print(f"Benchmark results for {model_name}:")
        print(f"  Average inference time: {avg_time:.2f} ms")
        print(f"  FPS: {fps:.2f}")
        print(f"  Model size: {model_size_mb:.2f} MB")
        
        return results
    
    def benchmark_onnx(self, onnx_path: str, model_name: str, num_runs: int = 100) -> Dict:
        """Benchmark ONNX model performance"""
        print(f"Benchmarking ONNX {model_name}...")
        
        # Create ONNX Runtime session
        session = ort.InferenceSession(onnx_path)
        
        # Create dummy input
        dummy_input = np.random.randn(self.batch_size, 3, self.img_size, self.img_size).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            _ = session.run(None, {'input': dummy_input})
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = session.run(None, {'input': dummy_input})
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1000 / avg_time
        
        # Get model size
        model_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        
        results = {
            'model_name': f"{model_name}_onnx",
            'avg_inference_time_ms': avg_time,
            'std_inference_time_ms': std_time,
            'fps': fps,
            'model_size_mb': model_size_mb,
            'num_runs': num_runs
        }
        
        print(f"ONNX benchmark results for {model_name}:")
        print(f"  Average inference time: {avg_time:.2f} ms")
        print(f"  FPS: {fps:.2f}")
        print(f"  Model size: {model_size_mb:.2f} MB")
        
        return results
    
    def export_model_complete(self, model_type: str, checkpoint_path: str, 
                            calibration_loader=None) -> Dict:
        """Complete export pipeline for a model"""
        print(f"Starting complete export for {model_type}...")
        
        # Load model
        model = self.load_model(model_type, checkpoint_path)
        
        results = {
            'model_type': model_type,
            'checkpoint_path': checkpoint_path,
            'exports': {},
            'benchmarks': {}
        }
        
        # Export TorchScript
        ts_path = self.export_torchscript(model, model_type)
        results['exports']['torchscript'] = ts_path
        
        # Export ONNX
        onnx_path = self.export_onnx(model, model_type)
        results['exports']['onnx'] = onnx_path
        
        # Benchmark original model
        orig_benchmark = self.benchmark_model(model, f"{model_type}_original")
        results['benchmarks']['original'] = orig_benchmark
        
        # Benchmark ONNX model
        onnx_benchmark = self.benchmark_onnx(onnx_path, model_type)
        results['benchmarks']['onnx'] = onnx_benchmark
        
        # Dynamic quantization
        try:
            quantized_model = self.quantize_dynamic(model, model_type)
            quant_benchmark = self.benchmark_model(quantized_model, f"{model_type}_quantized")
            results['benchmarks']['quantized'] = quant_benchmark
            
            # Export quantized TorchScript
            quant_ts_path = self.export_torchscript(quantized_model, f"{model_type}_quantized")
            results['exports']['torchscript_quantized'] = quant_ts_path
        except Exception as e:
            print(f"Dynamic quantization failed: {e}")
        
        # Static quantization (if calibration data provided)
        if calibration_loader is not None:
            try:
                static_quantized = self.quantize_static(model, model_type, calibration_loader)
                static_benchmark = self.benchmark_model(static_quantized, f"{model_type}_static_quantized")
                results['benchmarks']['static_quantized'] = static_benchmark
                
                # Export static quantized TorchScript
                static_ts_path = self.export_torchscript(static_quantized, f"{model_type}_static_quantized")
                results['exports']['torchscript_static_quantized'] = static_ts_path
            except Exception as e:
                print(f"Static quantization failed: {e}")
        
        # Save results
        results_path = self.export_dir / f"{model_type}_export_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Export results saved to: {results_path}")
        return results
    
    def export_all_models(self, checkpoint_dir: str, calibration_loader=None) -> Dict:
        """Export all available models"""
        checkpoint_dir = Path(checkpoint_dir)
        all_results = {}
        
        # Define model checkpoints to export
        model_checkpoints = {
            'gradcam': checkpoint_dir / 'gradcam_best.pth',
            'patch': checkpoint_dir / 'patch_forensics_best.pth',
            'attention': checkpoint_dir / 'attention_best.pth',
            'mobilenetv3': checkpoint_dir / 'mobilenetv3_best.pth'
        }
        
        for model_type, checkpoint_path in model_checkpoints.items():
            if checkpoint_path.exists():
                print(f"\n{'='*50}")
                print(f"Exporting {model_type.upper()} model")
                print(f"{'='*50}")
                
                try:
                    results = self.export_model_complete(
                        model_type, 
                        str(checkpoint_path), 
                        calibration_loader
                    )
                    all_results[model_type] = results
                except Exception as e:
                    print(f"Failed to export {model_type}: {e}")
                    all_results[model_type] = {'error': str(e)}
            else:
                print(f"Checkpoint not found for {model_type}: {checkpoint_path}")
        
        # Save combined results
        combined_results_path = self.export_dir / 'all_models_export_results.json'
        with open(combined_results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nAll export results saved to: {combined_results_path}")
        return all_results

def main():
    """Main function for model export"""
    config = Config()
    exporter = MobileExporter(config)
    
    # Export all models
    checkpoint_dir = config.CHECKPOINT_DIR
    results = exporter.export_all_models(checkpoint_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPORT SUMMARY")
    print("="*60)
    
    for model_type, result in results.items():
        if 'error' not in result:
            print(f"\n{model_type.upper()}:")
            for benchmark_type, benchmark in result['benchmarks'].items():
                print(f"  {benchmark_type}: {benchmark['avg_inference_time_ms']:.2f}ms, "
                      f"{benchmark['fps']:.1f} FPS, {benchmark['model_size_mb']:.2f}MB")
        else:
            print(f"\n{model_type.upper()}: FAILED - {result['error']}")

if __name__ == "__main__":
    main() 