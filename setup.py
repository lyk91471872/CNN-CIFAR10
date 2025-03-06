from setuptools import setup, find_packages

setup(
    name="cnn-cifar10",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "matplotlib",
        "tqdm",
        "pillow",
    ],
    description="CNN implementation for CIFAR-10 dataset",
    author="Your Name",
) 