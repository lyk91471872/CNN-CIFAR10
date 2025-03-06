from setuptools import setup, find_packages


def get_requirements():
    """Read requirements from requirements.txt file."""
    try:
        with open('requirements.txt') as f:
            return [
                line.strip() 
                for line in f 
                if line.strip() and not line.startswith('#')
            ]
    except FileNotFoundError:
        print("Warning: requirements.txt not found")
        return []


setup(
    name="cnn-cifar10",
    version="0.1.0",
    packages=find_packages(),
    install_requires=get_requirements(),
    description="CNN implementation for CIFAR-10 dataset",
    author="Your Name",
    python_requires='>=3.6',
) 