# CNN-CIFAR10

A PyTorch implementation of Convolutional Neural Networks for the CIFAR-10 dataset. This project provides a flexible and modular framework for training, evaluating, and deploying CNN models on the CIFAR-10 image classification task.

## Features

- **Multiple CNN Architectures**
  - ResNet18 - Residual Network with skip connections
  - EfficientNetV2-B0 - Efficient architecture with compound scaling
  - Extensible base model for adding new architectures

- **Advanced Training Capabilities**
  - Cross-validation support with averaged metrics visualization
  - Early stopping to prevent overfitting
  - Learning rate scheduling
  - Mixup augmentation for better generalization

- **Data Management**
  - Efficient data loading with optimized DataLoader
  - Comprehensive data augmentation pipeline with AutoAugment
  - Unified dataset interface with configurable modes
  - Benchmark tools for DataLoader performance optimization

- **Experiment Tracking**
  - SQLite database for tracking model runs, cross-validation results, and predictions
  - Automatic model versioning with timestamps
  - Training history visualization
  - Relationship tracking between models and predictions

- **Utility Scripts**
  - DataLoader benchmarking for performance optimization
  - Test set visualization with PDF export
  - Normalization value computation and config updates
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
│   ├── testset2pdf.py    # Export test set images to PDF
│   └── update_normalization_values.py # Update normalization values in config
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

### Command-Line Options

The main.py script provides several operation modes through command-line arguments:

| Option | Description |
|--------|-------------|
| `-t, --train` | Train the model on full dataset |
| `-c, --crossval` | Run cross-validation |
| `-p, --pdf` | Generate PDF of test images |
| `-b, --benchmark` | Run dataloader benchmark |
| `-n, --normalize` | Update normalization values |

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
- Create a training history visualization

### Cross-validation

To run cross-validation:

```bash
python main.py -c
```

This will:
- Split the dataset into k folds (default: 5)
- Train and evaluate the model on each fold
- Report the average performance across all folds
- Generate an averaged training history plot
- Record detailed cross-validation results in the database

### Generating a PDF of Test Images

To visualize the test dataset:

```bash
python main.py -p
```

This will create a PDF with all test images, organized in a grid with their IDs.

### Benchmarking the DataLoader

To find the optimal DataLoader configuration:

```bash
python main.py -b
```

This will test various combinations of batch sizes and worker counts to find the most efficient configuration for your hardware.

### Updating Normalization Values

To compute and update normalization values in config.py:

```bash
python main.py -n
```

This calculates the mean and standard deviation from the training dataset and updates the normalization values in the configuration.

## Configuration

All configuration options are centralized in `config.py`, including:

- Model selection
- DataLoader parameters
- Optimizer settings
- Normalization values
- Transform configurations
- Directory paths

## Database

The project includes a SQLite database (`models.db`) that tracks:

1. **Model Runs**
   - Model type
   - Weights file path
   - Timestamp
   - Configuration used

2. **Cross-validation Results**
   - Model type
   - Number of folds
   - Average validation accuracy and standard deviation
   - Path to the averaged history plot
   - Detailed fold results
   - Timestamp

3. **Predictions**
   - Link to the model run
   - Prediction file path
   - Timestamp

This allows for comprehensive tracking of experiments, providing a history of which models performed best and under what configurations.

## License

This project is licensed under the MIT License - see the LICENSE file for details.