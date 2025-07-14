#!/usr/bin/env python3
"""
Quick Installation Script for Deepfake Detection System
Handles all dependencies with proper error handling
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_dependencies():
    """Install all dependencies"""
    print("🚀 Installing Deepfake Detection System Dependencies")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Upgrade pip and setuptools
    if not run_command("pip install --upgrade pip setuptools wheel", "Upgrading pip and setuptools"):
        return False
    
    # Install PyTorch with CUDA support
    if not run_command("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118", 
                      "Installing PyTorch with CUDA support"):
        return False
    
    # Install core dependencies
    core_packages = [
        "numpy", "Pillow", "opencv-python", "tqdm",
        "fastapi", "uvicorn",
        "scikit-learn", "scikit-image", "matplotlib", "seaborn", "scipy",
        "numba", "transformers"
    ]
    
    for package in core_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️ Warning: {package} installation failed, continuing...")
    
    # Install XAI tools with special handling
    print("\n🔍 Installing XAI tools (latest versions)...")
    
    # Install CLIP from GitHub
    if not run_command("pip install git+https://github.com/openai/CLIP.git", "Installing CLIP"):
        print("⚠️ Warning: CLIP installation failed, trying alternative...")
        run_command("pip install ftfy regex", "Installing CLIP dependencies")
        run_command("pip install git+https://github.com/openai/CLIP.git", "Installing CLIP (retry)")
    
    # Install grad-cam
    if not run_command("pip install grad-cam", "Installing grad-cam"):
        print("⚠️ Warning: grad-cam failed, trying pytorch-grad-cam...")
        run_command("pip install pytorch-grad-cam", "Installing pytorch-grad-cam")
    
    # Install LIME
    if not run_command("pip install lime", "Installing LIME"):
        print("⚠️ Warning: LIME installation failed")
    
    # Install SHAP
    if not run_command("pip install shap", "Installing SHAP"):
        print("⚠️ Warning: SHAP installation failed")
    
    return True

def verify_installation():
    """Verify that all packages can be imported"""
    print("\n✅ Verifying installation...")
    
    packages_to_test = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("tqdm", "tqdm"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("sklearn", "scikit-learn"),
        ("skimage", "scikit-image"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("scipy", "scipy"),
        ("transformers", "transformers"),
        ("numba", "numba")
    ]
    
    failed_imports = []
    
    for package, name in packages_to_test:
        try:
            __import__(package)
            print(f"✅ {name} imported successfully")
        except ImportError:
            print(f"❌ {name} import failed")
            failed_imports.append(name)
    
    # Test XAI packages
    xai_packages = [
        ("pytorch_grad_cam", "Grad-CAM"),
        ("lime", "LIME"),
        ("shap", "SHAP"),
        ("clip", "CLIP")
    ]
    
    print("\n🔍 Testing XAI packages...")
    for package, name in xai_packages:
        try:
            __import__(package)
            print(f"✅ {name} imported successfully")
        except ImportError:
            print(f"⚠️ {name} import failed (optional)")
    
    # Test CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠️ CUDA not available, using CPU")
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
    
    if failed_imports:
        print(f"\n⚠️ Some core packages failed to import: {', '.join(failed_imports)}")
        return False
    
    print("\n🎉 Installation verification completed!")
    return True

def main():
    """Main installation function"""
    print("🎯 Deepfake Detection System - Quick Installation")
    print("=" * 60)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Installation failed. Please check the errors above.")
        return False
    
    # Verify installation
    if not verify_installation():
        print("\n⚠️ Some packages failed to install. Please check the warnings above.")
    
    print("\n" + "=" * 60)
    print("📋 INSTALLATION SUMMARY")
    print("=" * 60)
    print("✅ Core dependencies installed")
    print("✅ PyTorch with CUDA support installed")
    print("✅ XAI tools installed (with fallbacks)")
    print("\n🚀 Next steps:")
    print("1. Test the system: python test_core.py")
    print("2. Prepare your dataset in data/wacv_data/")
    print("3. Start training: python main.py --datasets celebahq")
    print("\n📖 For detailed installation guide, see INSTALL.md")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 