#!/usr/bin/env python3
# check_requirements.py - Verify all dependencies are installed

import sys
import importlib

def check_import(module_name, package_name=None):
    """Check if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {package_name or module_name}")
        return True
    except ImportError:
        print(f"❌ {package_name or module_name} - NOT INSTALLED")
        return False

def main():
    print("🔍 Checking MajorSignL Dependencies")
    print("=" * 40)
    
    all_good = True
    
    # Core dependencies
    all_good &= check_import("cv2", "opencv")
    all_good &= check_import("numpy")
    all_good &= check_import("mediapipe")
    all_good &= check_import("face_recognition")
    all_good &= check_import("sklearn", "scikit-learn")
    all_good &= check_import("fastapi")
    all_good &= check_import("uvicorn")
    all_good &= check_import("websockets")
    
    # Try TensorFlow (for sign language)
    tf_available = check_import("tensorflow")
    
    # Try PyTorch (for potential GPU acceleration)
    torch_available = check_import("torch", "pytorch")
    
    if torch_available:
        try:
            import torch
            print(f"  🔥 PyTorch version: {torch.__version__}")
            print(f"  🚀 CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"  🎮 GPU: {torch.cuda.get_device_name(0)}")
        except:
            pass
    
    print("\n" + "=" * 40)
    
    if all_good:
        print("🎉 All core dependencies are installed!")
        print("You should be able to run the face recognition system.")
        
        if not tf_available:
            print("⚠️  TensorFlow not found - sign language classification won't work")
            print("   Install with: pip install tensorflow")
            
    else:
        print("❌ Some dependencies are missing!")
        print("Please install missing packages and try again.")
        print("\nTo install missing packages:")
        print("mamba activate SignL")
        print("pip install face_recognition dlib cmake")
        
    return all_good

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
