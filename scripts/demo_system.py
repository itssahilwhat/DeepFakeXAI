#!/usr/bin/env python3
"""
Deepfake Detection System Demo
Shows all components working without requiring a trained model
"""

import os
import sys
import logging
import torch
from src.config import Config
from src.model import EfficientNetLiteTemporal
from src.data import get_dataloader
from src.test_system import test_inference_speed, test_training_speed

def demo_model_creation():
    """Demo: Create and test model architecture"""
    logging.info("🔧 Testing model creation...")
    
    model = EfficientNetLiteTemporal(pretrained=False).to(Config.DEVICE)
    model.eval()
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, *Config.INPUT_SIZE).to(Config.DEVICE)
    with torch.no_grad():
        cls_output, seg_output = model(dummy_input)
    
    logging.info(f"✅ Model created successfully!")
    logging.info(f"   Classification output shape: {cls_output.shape}")
    logging.info(f"   Segmentation output shape: {seg_output.shape}")
    logging.info(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

def demo_data_loading():
    """Demo: Test data loading"""
    logging.info("📦 Testing data loading...")
    
    try:
        train_loader = get_dataloader("celebahq", "train", batch_size=4)
        batch = next(iter(train_loader))
        
        logging.info(f"✅ Data loading successful!")
        logging.info(f"   Batch keys: {list(batch.keys())}")
        logging.info(f"   Image shape: {batch['image'].shape}")
        logging.info(f"   Mask shape: {batch['mask'].shape}")
        logging.info(f"   Dataset size: {len(train_loader.dataset)}")
        
        return train_loader
    except Exception as e:
        logging.error(f"❌ Data loading failed: {e}")
        return None

def demo_speed_testing():
    """Demo: Test speed benchmarks"""
    logging.info("⚡ Testing speed benchmarks...")
    
    try:
        # Test inference speed
        inference_speed = test_inference_speed()
        logging.info(f"✅ Inference speed test: {inference_speed:.1f} it/sec")
        
        # Test training speed
        train_loader = get_dataloader("celebahq", "train", batch_size=Config.BATCH_SIZE)
        training_speed = test_training_speed()
        logging.info(f"✅ Training speed test: {training_speed:.1f} it/sec")
        
        return True
    except Exception as e:
        logging.error(f"❌ Speed testing failed: {e}")
        return False

def demo_visualization():
    """Demo: Test visualization system"""
    logging.info("🎨 Testing visualization system...")
    
    try:
        from src.visualization import generate_all_visualizations
        generate_all_visualizations()
        logging.info("✅ Visualization system working!")
        return True
    except Exception as e:
        logging.error(f"❌ Visualization failed: {e}")
        return False

def demo_onnx_export():
    """Demo: Test ONNX export (without trained model)"""
    logging.info("📤 Testing ONNX export system...")
    
    try:
        # Create a dummy model for export testing
        model = EfficientNetLiteTemporal(pretrained=False).to(Config.DEVICE)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, *Config.INPUT_SIZE).to(Config.DEVICE)
        
        # Test ONNX export
        os.makedirs(Config.ONNX_DIR, exist_ok=True)
        onnx_path = os.path.join(Config.ONNX_DIR, "demo_model.onnx")
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
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
        
        logging.info(f"✅ ONNX export successful: {onnx_path}")
        return True
    except Exception as e:
        logging.error(f"❌ ONNX export failed: {e}")
        return False

def demo_xai_components():
    """Demo: Test XAI components"""
    logging.info("🧠 Testing XAI components...")
    
    try:
        # Test GradCAM
        from src.utils import generate_gradcam
        model = EfficientNetLiteTemporal(pretrained=False).to(Config.DEVICE)
        dummy_input = torch.randn(1, 3, *Config.INPUT_SIZE).to(Config.DEVICE)
        
        gradcam_result = generate_gradcam(model, dummy_input)
        logging.info(f"✅ GradCAM working: {gradcam_result.shape}")
        
        # Test LIME explainer
        from src.lime_explainer import LIMEDeepfakeExplainer
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize(Config.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        lime_explainer = LIMEDeepfakeExplainer(model, transform)
        logging.info("✅ LIME explainer created successfully")
        
        return True
    except Exception as e:
        logging.error(f"❌ XAI components failed: {e}")
        return False

def main():
    """Run complete system demo"""
    Config.setup_logging()
    logging.info("🎯 Starting Deepfake Detection System Demo")
    logging.info("="*60)
    
    results = {}
    
    # Test each component
    results['Model Creation'] = True if demo_model_creation() else False
    results['Data Loading'] = True if demo_data_loading() else False
    results['Speed Testing'] = demo_speed_testing()
    results['Visualization'] = demo_visualization()
    results['ONNX Export'] = demo_onnx_export()
    results['XAI Components'] = demo_xai_components()
    
    # Summary
    logging.info("="*60)
    logging.info("DEMO RESULTS SUMMARY")
    logging.info("="*60)
    
    success_count = sum(results.values())
    total_count = len(results)
    
    for component, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logging.info(f"{component}: {status}")
    
    logging.info(f"\nOverall: {success_count}/{total_count} components working")
    
    if success_count == total_count:
        logging.info("🎉 ALL SYSTEMS OPERATIONAL!")
        logging.info("Your deepfake detection system is ready for training and deployment!")
    else:
        logging.warning("⚠️ Some components need attention. Check logs above.")
    
    return success_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 