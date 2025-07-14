import torch
import numpy as np
from src.model import EfficientNetLiteTemporal
from src.interpretability import InterpretabilityTools
from src.config import Config

def debug_gradcam():
    print("🔍 Debugging GradCAM...")
    
    # Create model
    model = EfficientNetLiteTemporal(pretrained=False).to(Config.DEVICE)
    model.eval()
    
    # Print model structure
    print("\n📋 Model structure:")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print(f"  {name}: {module}")
    
    # Create interpretability tools
    interpretability = InterpretabilityTools(model, Config.DEVICE)
    
    # Test target layer finding
    print(f"\n🎯 Looking for target layer: {Config.GRADCAM_TARGET_LAYER}")
    target_layer = interpretability._find_target_layer()
    print(f"Found target layer: {target_layer}")
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 256, 256).to(Config.DEVICE)
    
    # Test GradCAM
    try:
        gradcam = interpretability.generate_gradcam(dummy_input)
        if gradcam is not None:
            print(f"✅ GradCAM successful! Shape: {gradcam.shape}")
        else:
            print("❌ GradCAM returned None")
    except Exception as e:
        print(f"❌ GradCAM failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_gradcam() 