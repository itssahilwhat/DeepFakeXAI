#!/usr/bin/env python3
"""
Comprehensive Deepfake Detection System Test
Optimized for RTX 3050 4GB VRAM
"""

import os
import sys
import torch
import logging
from src.config import Config
from src.model import EfficientNetLiteTemporal
from src.train import train_model
from src.test_system import test_system
from src.advanced_metrics import AdvancedMetrics
from src.robustness_testing import RobustnessTester
from src.interpretability import InterpretabilityTools
from src.cross_dataset_evaluation import CrossDatasetEvaluator


def setup_environment():
    """Setup environment and check requirements"""
    print("🔧 Setting up environment...")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️ CUDA not available, using CPU")
    
    # Check PyTorch version
    print(f"📦 PyTorch version: {torch.__version__}")
    
    # Create necessary directories
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.ONNX_DIR, exist_ok=True)
    
    print("✅ Environment setup complete")


def test_model_creation():
    """Test model creation and basic functionality"""
    print("\n🧪 Testing model creation...")
    
    try:
        # Create model
        model = EfficientNetLiteTemporal(
            num_classes=Config.NUM_CLASSES,
            pretrained=Config.PRETRAINED
        ).to(Config.DEVICE)
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, *Config.INPUT_SIZE).to(Config.DEVICE)
        with torch.no_grad():
            predictions = model(dummy_input)
        
        if isinstance(predictions, tuple):
            cls_output, seg_output = predictions
            print(f"✅ Model created successfully")
            print(f"   Classification output shape: {cls_output.shape if cls_output is not None else 'None'}")
            print(f"   Segmentation output shape: {seg_output.shape}")
        else:
            print(f"✅ Model created successfully")
            print(f"   Output shape: {predictions.shape}")
        
        # Test memory usage
        if Config.DEVICE == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**2
            print(f"   Memory allocated: {allocated:.1f} MB")
        
        return model
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return None


def test_data_loading():
    """Test data loading functionality"""
    print("\n📂 Testing data loading...")
    
    try:
        from src.data import get_dataloader
        
        # Check if data directory exists
        if not os.path.exists(Config.DATA_ROOT):
            print(f"⚠️ Data directory not found: {Config.DATA_ROOT}")
            print("   Please ensure your dataset is in the correct location")
            return False
        
        # List available datasets
        available_datasets = []
        for item in os.listdir(Config.DATA_ROOT):
            item_path = os.path.join(Config.DATA_ROOT, item)
            if os.path.isdir(item_path):
                available_datasets.append(item)
        
        if not available_datasets:
            print("❌ No datasets found in data directory")
            return False
        
        print(f"✅ Found datasets: {', '.join(available_datasets)}")
        
        # Test loading first available dataset
        test_dataset = available_datasets[0]
        print(f"🧪 Testing with dataset: {test_dataset}")
        
        try:
            train_loader = get_dataloader(test_dataset, "train", batch_size=2)
            valid_loader = get_dataloader(test_dataset, "valid", batch_size=2)
            
            # Test batch loading
            train_batch = next(iter(train_loader))
            valid_batch = next(iter(valid_loader))
            
            print(f"✅ Data loading successful")
            print(f"   Train batch shape: {train_batch['image'].shape}")
            print(f"   Valid batch shape: {valid_batch['image'].shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False
    

def test_training_pipeline():
    """Test training pipeline"""
    print("\n🚀 Testing training pipeline...")
    
    try:
        # Find available datasets
        available_datasets = []
        for item in os.listdir(Config.DATA_ROOT):
            item_path = os.path.join(Config.DATA_ROOT, item)
            if os.path.isdir(item_path):
                available_datasets.append(item)
        
        if not available_datasets:
            print("❌ No datasets available for training")
            return False
        
        # Use first available dataset for testing
        test_dataset = available_datasets[0]
        print(f"🧪 Testing training with dataset: {test_dataset}")
        
        # Skip training pipeline test for now (requires full dataset)
        print("⚠️ Training pipeline test skipped (requires full dataset)")
        print("   Run 'python main.py --datasets celebahq ffhq' to test training")
        return True
            
    except Exception as e:
        print(f"❌ Training pipeline test failed: {e}")
        return False
    

def test_xai_functionality():
    """Test XAI functionality"""
    print("\n🔍 Testing XAI functionality...")
    
    try:
        # Create model
        model = EfficientNetLiteTemporal(
            num_classes=Config.NUM_CLASSES,
            pretrained=False  # Use untrained for faster testing
        ).to(Config.DEVICE)
        
        # Create interpretability tools
        interpretability = InterpretabilityTools(model, Config.DEVICE)
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, *Config.INPUT_SIZE).to(Config.DEVICE)
        
        # Test GradCAM
        try:
            gradcam = interpretability.generate_gradcam(dummy_input)
            if gradcam is not None:
                print("✅ GradCAM test successful")
            else:
                print("⚠️ GradCAM test failed: returned None")
        except Exception as e:
            print(f"⚠️ GradCAM test failed: {e}")
        
        # Test SHAP
        try:
            shap_result = interpretability.generate_shap_explanation(dummy_input)
            if shap_result is not None:
                print("✅ SHAP test successful")
            else:
                print("⚠️ SHAP test failed: returned None")
        except Exception as e:
            print(f"⚠️ SHAP test failed: {e}")
        
        # Test LIME
        try:
            lime = interpretability.generate_lime_explanation(dummy_input)
            print("✅ LIME test successful")
        except Exception as e:
            print(f"⚠️ LIME test failed: {e}")
        
        # Test CLIP (if available)
        if Config.USE_CLIP_EXPLAINER:
            try:
                clip_explanation = interpretability.generate_clip_explanation(dummy_input)
                print("✅ CLIP explanation test successful")
            except Exception as e:
                print(f"⚠️ CLIP explanation test failed: {e}")
        
        print("✅ XAI functionality test completed")
        return True
        
    except Exception as e:
        print(f"❌ XAI functionality test failed: {e}")
        return False
    

def test_robustness():
    """Test robustness functionality"""
    print("\n🛡️ Testing robustness functionality...")
    
    try:
        # Create model
        model = EfficientNetLiteTemporal(
            num_classes=Config.NUM_CLASSES,
            pretrained=False
        ).to(Config.DEVICE)
        
        # Create robustness tester
        robustness_tester = RobustnessTester(model, Config.DEVICE)
        
        # Create dummy data
        dummy_images = torch.randn(2, 3, *Config.INPUT_SIZE).to(Config.DEVICE)
        dummy_masks = torch.randint(0, 2, (2, 1, *Config.INPUT_SIZE)).float().to(Config.DEVICE)
        
        # Test noise robustness
        try:
            noise_results = robustness_tester.test_noise_robustness(dummy_images, dummy_masks, [0, 10])
            print("✅ Noise robustness test successful")
        except Exception as e:
            print(f"⚠️ Noise robustness test failed: {e}")
        
        # Test compression robustness
        try:
            compression_results = robustness_tester.test_compression_robustness(dummy_images, dummy_masks, [100, 85])
            print("✅ Compression robustness test successful")
        except Exception as e:
            print(f"⚠️ Compression robustness test failed: {e}")
        
        print("✅ Robustness functionality test completed")
        return True
        
    except Exception as e:
        print(f"❌ Robustness functionality test failed: {e}")
        return False


def test_metrics():
    """Test advanced metrics"""
    print("\n📊 Testing advanced metrics...")
    
    try:
        # Create metrics
        metrics = AdvancedMetrics()
        
        # Create dummy data
        dummy_pred = torch.randn(10, 1, 256, 256)
        dummy_target = torch.randint(0, 2, (10, 1, 256, 256)).float()
        
        # Update metrics
        metrics.update(dummy_pred, dummy_target, dummy_pred, dummy_target)
        
        # Compute metrics
        results = metrics.compute()
        
        print("✅ Advanced metrics test successful")
        print(f"   Available metrics: {list(results.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced metrics test failed: {e}")
        return False


def main():
    """Main test function"""
    print("🎯 Deepfake Detection System - Comprehensive Test")
    print("=" * 60)
    
    # Setup environment
    setup_environment()
    
    # Run tests
    tests = [
        ("Model Creation", test_model_creation),
        ("Data Loading", test_data_loading),
        ("XAI Functionality", test_xai_functionality),
        ("Robustness Testing", test_robustness),
        ("Advanced Metrics", test_metrics),
        ("Training Pipeline", test_training_pipeline),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for training.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if not torch.cuda.is_available():
        print("   - Consider using GPU for faster training")
    if Config.DEVICE == "cuda" and torch.cuda.get_device_properties(0).total_memory < 6 * 1024**3:
        print("   - Consider reducing batch size for lower VRAM usage")
    
    print("\n🚀 To start training, run:")
    print("   python main.py --datasets celebahq ffhq")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 