"""
Setup script for Qwen 2.5VL OCR environment
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, shell=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor} is compatible")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor} is not compatible. Requires Python 3.8+")
        return False

def setup_environment():
    """Setup the OCR environment"""
    print("="*50)
    print("Qwen 2.5VL OCR Environment Setup")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check if virtual environment is activated
    venv_path = Path(".venv")
    if venv_path.exists():
        print("✓ Virtual environment found")
    else:
        print("Creating virtual environment...")
        if not run_command("python -m venv .venv", "Creating virtual environment"):
            return False
    
    # Activate virtual environment (Windows)
    activate_script = ".venv\\Scripts\\Activate.ps1"
    if not Path(activate_script).exists():
        print("✗ Virtual environment activation script not found")
        return False
    
    print("Virtual environment ready. To activate manually:")
    print(f"  .\\{activate_script}")
    
    # Install requirements
    pip_cmd = ".venv\\Scripts\\pip.exe"
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        return False
    
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing requirements"):
        return False
    
    # Create input/output directories
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    print("✓ Created input and output directories")
    
    print("\n" + "="*50)
    print("Setup completed successfully!")
    print("="*50)
    print("\nNext steps:")
    print("1. Activate virtual environment: .\\venv\\Scripts\\Activate.ps1")
    print("2. Place your images in the 'input' folder")
    print("3. Run: python example_usage.py")
    print("\nFor help: python batch_ocr.py --help")
    
    return True

if __name__ == "__main__":
    try:
        success = setup_environment()
        if not success:
            print("\nSetup failed. Please check the errors above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nSetup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during setup: {e}")
        sys.exit(1)
