# CNN-CIFAR10

A PyTorch implementation of Convolutional Neural Networks for the CIFAR-10 dataset. This project provides a flexible and modular framework for training, evaluating, and deploying CNN models on the CIFAR-10 image classification task.

## Features

- **Multiple CNN Architectures**
  - ResNet18 - Residual Network with skip connections
  - EfficientNetV2-B0 - Efficient architecture with compound scaling
  - Extensible base model for adding new architectures

- **Advanced Training Capabilities**
  - Cross-validation support
  - Early stopping to prevent overfitting
  - Learning rate scheduling
  - Mixup augmentation for better generalization
  - Progressive learning with increasing augmentation probability

- **Data Management**
  - Efficient data loading with optimized DataLoader
  - Comprehensive data augmentation pipeline
  - Support for both training and test datasets
  - Benchmark tools for DataLoader performance optimization

- **Experiment Tracking**
  - SQLite database for tracking model runs and predictions
  - Automatic model versioning with timestamps
  - Training history visualization
  - Relationship tracking between models and predictions

- **Utility Scripts**
  - DataLoader benchmarking for performance optimization
  - Test set visualization with PDF export
  - Organized outputs for all scripts

## Project Structure

```
CNN-CIFAR10/
├── __init__.py           # Root package initialization
├── config.py             # Global configuration settings
├── dataset.py            # Dataset classes for CIFAR-10
├── main.py               # Main script for training and prediction
├── models/               # Model definitions
│   ├── __init__.py       # Models package initialization
│   ├── base.py           # Base model class with save/load functionality
│   ├── resnet.py         # ResNet-18 implementation
│   └── efficientnet.py   # EfficientNetV2-B0 implementation
├── scripts/              # Utility scripts
│   ├── __init__.py       # Scripts package initialization
│   ├── dataloader_benchmark.py  # Benchmark dataloader performance
│   ├── outputs/          # Directory for script outputs
│   └── testset2pdf.py    # Export test set images to PDF
├── utils/                # Utility modules
│   ├── __init__.py       # Utils package initialization
│   ├── augmentation.py   # Data augmentation utilities
│   ├── db.py             # Database utilities for tracking experiments
│   ├── early_stopping.py # Early stopping implementation
│   ├── pipeline.py       # Training and evaluation pipeline
│   └── visualization.py  # Training history visualization
├── weights/              # Model weights storage
├── graphs/               # Training graphs output
└── predictions/          # Model predictions storage
```

## Installation

### Prerequisites

- Python 3.6+
- PyTorch 1.7+
- CUDA-capable GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CNN-CIFAR10.git
cd CNN-CIFAR10
```

2. Install the package in development mode:
```bash
pip install -e .
```

This will automatically install all dependencies from `requirements.txt`.

## Usage

### Training a Model

To train a model on the full dataset:

```bash
python main.py -t
```

This will:
- Load the CIFAR-10 dataset
- Initialize the model specified in `config.py`
- Train the model with the configured parameters
- Save the best model weights
- Generate a timestamped prediction file
- Record the model and prediction in the database

### Cross-validation

To run cross-validation:

```bash
python main.py -c
```

This will:
- Split the dataset into k folds (default: 5)
- Train and evaluate the model on each fold
- Report the average performance across all folds

### Running Utility Scripts

#### DataLoader Benchmark

To find the optimal DataLoader configuration:

```bash
python scripts/dataloader_benchmark.py
```

This will test various combinations of batch sizes and worker counts to find the most efficient configuration for your hardware.

#### Test Set Visualization

To generate a PDF of the test set images:

```bash
python scripts/testset2pdf.py
```

This will create a PDF with all test images, organized in a grid with their IDs.

## Configuration

All configuration options are centralized in `config.py`:

### Model Selection
```python
MODEL = CustomEfficientNetV2_B0  # or CustomResNet18
```

### Training Parameters
```python
TRAIN = {
    'epochs': 100,
    'early_stopping_patience': 10,
    'mixup_alpha': 0.2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}
```

### Data Augmentation
```python
TRANSFORM = transforms.Compose([
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10)
])
```

### DataLoader Settings
```python
DATALOADER = {
    'batch_size': 512,
    'num_workers': 32,
    'pin_memory': True,
}
```

## Database

The project includes a SQLite database (`models.db`) that tracks:

1. **Model Runs**
   - Model type
   - Weights file path
   - Timestamp
   - Configuration used

2. **Predictions**
   - Link to the model run
   - Prediction file path
   - Timestamp

This allows for easy tracking of which model produced which predictions and under what configuration.

## Progressive Learning

The project supports progressive learning, where data augmentation probability increases over time:

```python
# In config.py
TRAIN = {
    # ...
    'progressive_learning': True,
    'aug_start_prob': 0.0,  # Starting probability
    'aug_end_prob': 1.0,    # Ending probability
    'aug_ramp_epochs': 20,  # Epochs to ramp up
}
```

This gradually introduces augmentation, allowing the model to learn from clean data first before adapting to augmented examples.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
