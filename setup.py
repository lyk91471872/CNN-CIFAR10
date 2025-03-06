from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="cnn-cifar10",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    description="CNN implementation for CIFAR-10 dataset",
    author="Your Name",
    python_requires='>=3.6',
) 