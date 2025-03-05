import subprocess
import sys
import re

def install_requirements():
    """Check and install required packages from requirements.txt silently."""
    print("Checking requirements...")
    try:
        with open('requirements.txt', 'r') as f:
            required_packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        raise Exception("requirements.txt not found")
    
    for package in required_packages:
        # Extract package name without version specifier
        package_name = re.split(r'[<>=~!]', package)[0].strip()
        try:
            __import__(package_name)
        except ImportError:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])
            except subprocess.CalledProcessError as e:
                raise Exception(f"Failed to install {package}: {e}") 