# CNN-CIFAR10

A PyTorch implementation of Convolutional Neural Networks for the CIFAR-10 dataset.

## Installation

To install the package in development mode:

```bash
pip install -e .
```

Alternatively, you can check for missing dependencies using the provided script:

```bash
./check_dependencies.sh
```

This interactive script will:
1. Check which dependencies are already installed
2. Show you any missing dependencies
3. Offer to install them automatically

## Project Structure

```
CNN-CIFAR10/
├── __init__.py           # Root package initialization
├── config.py             # Global configuration settings
├── dataset.py            # Dataset classes for CIFAR-10
├── main.py               # Main script for training and prediction
├── models/               # Model definitions
│   ├── __init__.py       # Models package initialization
│   ├── base.py           # Base model class
│   ├── resnet.py         # ResNet-18 implementation
│   └── efficientnet.py   # EfficientNet implementation
├── scripts/              # Utility scripts
│   ├── __init__.py       # Scripts package initialization
│   ├── dataloader_benchmark.py  # Benchmark dataloader performance
│   ├── outputs/          # Directory for script outputs
│   └── testset2pdf.py    # Export test set images to PDF
├── utils/                # Utility modules
│   ├── __init__.py       # Utils package initialization
│   ├── augmentation.py   # Data augmentation utilities
│   ├── db.py             # Database utilities
│   ├── pipeline.py       # Training and evaluation pipeline
│   └── visualization.py  # Visualization utilities
├── weights/              # Model weights
├── graphs/               # Training graphs
└── predictions/          # Model predictions
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

### Training a model

```bash
python main.py -t
```

### Cross-validation

```bash
python main.py -c
```

### Running scripts

```bash
# Run dataloader benchmark
python scripts/dataloader_benchmark.py

# Generate test set PDF
python scripts/testset2pdf.py
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

## Database

The project includes a SQLite database (`models.db`) that tracks relationships between:
- Model weights
- Prediction files
- Configuration settings

This allows for easy retrieval of which model produced which predictions.
