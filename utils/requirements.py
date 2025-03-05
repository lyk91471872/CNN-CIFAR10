import subprocess
import sys

def install_requirements():
    """Check and install required packages from requirements.txt."""
    try:
        with open('requirements.txt', 'r') as f:
            required_packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        print("requirements.txt not found. Skipping package installation.")
        return
    
    for package in required_packages:
        try:
            __import__(package.split('==')[0])  # Handle version specifiers
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])
            print(f"{package} installed successfully.") 