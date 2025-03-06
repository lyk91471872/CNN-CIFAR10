import subprocess
import sys
import os
import re

def install_requirements(check_only=False):
    """Check and install required packages from requirements.txt silently.
    
    Args:
        check_only: If True, only check if packages are installed without installing
    
    Environment variables:
        SKIP_REQUIREMENTS_CHECK: If set to '1', skip all requirements checks
    """
    # Skip check if environment variable is set
    if os.environ.get('SKIP_REQUIREMENTS_CHECK') == '1':
        print("Skipping requirements check (SKIP_REQUIREMENTS_CHECK=1)")
        return
        
    print("Checking requirements...")
    try:
        with open('requirements.txt', 'r') as f:
            required_packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        raise Exception("requirements.txt not found")
    
    missing_packages = []
    
    for package in required_packages:
        # Extract package name without version specifier
        package_name = re.split(r'[<>=~!]', package)[0].strip()
        try:
            __import__(package_name)
        except ImportError:
            if check_only:
                missing_packages.append(package)
            else:
                print(f"Installing {package}...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])
                except subprocess.CalledProcessError as e:
                    raise Exception(f"Failed to install {package}: {e}")
    
    if check_only and missing_packages:
        return missing_packages
    return [] 