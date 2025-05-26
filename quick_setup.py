"""
Quick test and setup script for Qwen 2.5VL OCR
"""

import os
import sys
from pathlib import Path

def create_folder_structure():
    """Create the test folder structure as specified"""
    print("Creating folder structure...")
    
    folders = [
        "input/folder1/subfolder1",
        "input/folder1/subfolder2", 
        "input/folder2",
        "output"
    ]
    
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {folder}")
    
    print("\nFolder structure created successfully!")
    print("\nPlease place your .tif images in:")
    print("- input/folder1/subfolder1/ (★ starred folder - PDF name)")
    print("- input/folder1/subfolder2/ (★ starred folder - PDF name)")  
    print("- input/folder2/ (★ starred folder - PDF name)")
    print("\nThe starred folders represent the original PDF names.")

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} imported successfully")
        
        import transformers
        print(f"✓ Transformers {transformers.__version__} imported successfully")
        
        import cv2
        print(f"✓ OpenCV {cv2.__version__} imported successfully")
        
        from PIL import Image
        print("✓ Pillow imported successfully")
        
        import numpy as np
        print(f"✓ NumPy {np.__version__} imported successfully")
        
        # Test device availability
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("ℹ CUDA not available - will use CPU")
            
        print(f"✓ CPU cores available: {os.cpu_count()}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def main():
    print("="*60)
    print("Qwen 2.5VL OCR - Quick Setup and Test")
    print("="*60)
    
    # Test imports first
    if not test_imports():
        print("\n❌ Package import test failed!")
        print("Please run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All packages imported successfully!")
    
    # Create folder structure
    print("\n" + "-"*40)
    create_folder_structure()
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Place your .tif images in the input folders")
    print("2. Run OCR processing:")
    print("   python batch_ocr.py input output")
    print("\nOr use the example script:")
    print("   python example_usage.py")
    print("\nFor help:")
    print("   python batch_ocr.py --help")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
