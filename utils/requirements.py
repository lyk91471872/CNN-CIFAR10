import subprocess
import sys

def install_requirements():
    """Check and install required packages from requirements.txt silently."""
    try:
        with open('requirements.txt', 'r') as f:
            required_packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        raise Exception("requirements.txt not found")
    
    for package in required_packages:
        try:
            __import__(package.split('==')[0])  # Handle version specifiers
        except ImportError:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])
            except subprocess.CalledProcessError as e:
                raise Exception(f"Failed to install {package}: {e}") 