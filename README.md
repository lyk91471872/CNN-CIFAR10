# CNN-CIFAR10

A PyTorch implementation of CNN models for CIFAR-10 image classification, with a focus on efficient data loading and training.

## Project Structure

```
CNN-CIFAR10/
├── config.py              # Global configuration settings
├── dataset.py            # CIFAR-10 dataset implementation
├── dataloader_benchmark.py # Benchmark script for DataLoader performance
├── main.py               # Main training script
├── models/               # Model architectures
│   ├── __init__.py
│   ├── base.py          # Base model class with save/load functionality
│   ├── efficientnet.py  # EfficientNetV2-B0 implementation
│   └── resnet.py        # ResNet18 implementation
├── utils/               # Utility functions
│   ├── __init__.py
│   ├── augmentation.py  # Data augmentation utilities
│   ├── early_stopping.py # Early stopping implementation
│   ├── pipeline.py      # Training pipeline
│   └── visualization.py # Training visualization utilities
├── weights/             # Saved model weights
└── graphs/             # Training visualization outputs
```

## Features

- Support for multiple CNN architectures (ResNet18, EfficientNetV2)
- Data augmentation:
  - Random horizontal flips
  - Random rotations
  - Color jittering
  - Mixup augmentation
- Training features:
  - Cross-validation support
  - Early stopping
  - Learning rate scheduling
  - Model checkpointing
- Performance optimization:
  - Efficient data loading with multiple workers
  - DataLoader benchmarking tool
  - GPU support with automatic device selection

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download CIFAR-10 dataset:
```bash
# The dataset will be automatically downloaded when running the training script
```

## Usage

### Training

```bash
python main.py
```

### DataLoader Benchmarking

```bash
python dataloader_benchmark.py
```

## Configuration

Key parameters can be modified in `config.py`:
- Model architecture
- Batch size
- Learning rate
- Number of workers
- Data augmentation settings
- Training parameters (epochs, early stopping, etc.)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
