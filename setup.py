#!/usr/bin/env python3
"""
Setup script for GRPO CPU Demo
Handles installation and environment configuration
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(command, check=True):
    """Run a command and handle errors"""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def install_pytorch_cpu():
    """Install CPU-only PyTorch"""
    print("🔧 Installing CPU-only PyTorch...")
    
    pytorch_install_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    
    if run_command(pytorch_install_cmd):
        print("✅ PyTorch CPU installed successfully")
        return True
    else:
        print("❌ Failed to install PyTorch")
        return False


def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    if run_command("pip install -r requirements.txt"):
        print("✅ Requirements installed successfully")
        return True
    else:
        print("❌ Failed to install requirements")
        return False


def verify_installation():
    """Verify that key packages are installed correctly"""
    print("🔍 Verifying installation...")
    
    packages_to_check = [
        "torch",
        "transformers", 
        "datasets",
        "trl",
        "gradio",
        "accelerate"
    ]
    
    all_good = True
    for package in packages_to_check:
        try:
            __import__(package)
            print(f"✅ {package} imported successfully")
        except ImportError:
            print(f"❌ {package} failed to import")
            all_good = False
    
    # Test PyTorch CPU setup
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print("⚠️  Warning: CUDA detected, but this demo is optimized for CPU")
        else:
            print("✅ Running on CPU (as intended)")
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        all_good = False
    
    return all_good


def setup_directories():
    """Create necessary directories"""
    print("📁 Setting up directories...")
    
    directories = [
        "grpo_output",
        "data",
        "logs",
        "checkpoints"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")


def test_basic_functionality():
    """Test basic functionality"""
    print("🧪 Testing basic functionality...")
    
    try:
        # Test imports
        from src.training.grpo_trainer import CPUGRPOTrainer, CPUGRPOConfig
        from src.utils.grpo_utils import RewardFunctions
        print("✅ Core modules imported successfully")
        
        # Test config creation
        config = CPUGRPOConfig()
        print("✅ Configuration created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def main():
    """Main setup function"""
    print("🚀 GRPO CPU Demo Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check platform
    system = platform.system()
    print(f"🖥️  Operating System: {system}")
    
    # Setup directories
    setup_directories()
    
    # Install PyTorch CPU
    if not install_pytorch_cpu():
        print("⚠️  PyTorch installation failed, trying to continue...")
    
    # Install other requirements
    if not install_requirements():
        print("❌ Requirements installation failed")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("❌ Installation verification failed")
        print("Please check the error messages above and try installing manually")
        sys.exit(1)
    
    # Test basic functionality
    if not test_basic_functionality():
        print("❌ Basic functionality test failed")
        print("Installation may be incomplete")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Launch the web interface: python app.py")
    print("2. Or try command-line training: python train_grpo.py --help")
    print("3. Open the tutorial notebook: notebooks/llm_rl_getting_started.ipynb")
    print("\n📖 For more information, see README.md")


if __name__ == "__main__":
    main()
