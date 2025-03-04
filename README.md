# CNN-CIFAR10

A PyTorch implementation of CNN models for CIFAR-10 image classification, with a focus on efficient data loading and training.

## Project Structure

- `main.py`: Main training script
- `config.py`: Global configuration settings
- `dataset.py`: CIFAR-10 dataset implementation
- `dataloader_benchmark.py`: Benchmark script for DataLoader performance
- `requirements.txt`: Project dependencies

## Features

- Support for multiple CNN architectures (ResNet18, EfficientNetV2)
- Automatic mixed precision training
- Mixup data augmentation
- Configurable DataLoader settings
- Performance benchmarking tools

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.
