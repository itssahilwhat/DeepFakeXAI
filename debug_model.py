import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def test_efficientnet():
    print("Testing EfficientNet B0 creation...")
    try:
        # Test basic EfficientNet
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        print("✅ Basic EfficientNet created successfully")
        
        # Test with dummy input
        dummy_input = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✅ EfficientNet forward pass successful, output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"❌ EfficientNet test failed: {e}")
        return False

def test_custom_model():
    print("\nTesting custom model creation...")
    try:
        from src.model import EfficientNetLiteTemporal
        model = EfficientNetLiteTemporal(pretrained=False)
        print("✅ Custom model created successfully")
        
        # Test with dummy input
        dummy_input = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✅ Custom model forward pass successful, output shape: {output[1].shape if isinstance(output, tuple) else output.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Custom model test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Debugging Model Creation Issues")
    print("=" * 50)
    
    # Test 1: Basic EfficientNet
    test_efficientnet()
    
    # Test 2: Custom model
    test_custom_model() 