import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
import datetime
from pathlib import Path

from models import CustomResNet18, CustomEfficientNetV2_B0

# Base directory paths
ROOT_DIR = Path(__file__).resolve().parent
DB_PATH = ROOT_DIR / 'models.db'

# Model selection
MODEL = CustomEfficientNetV2_B0

# Data paths
DATA_DIR = 'data/cifar-10-python/cifar-10-batches-py'
TRAIN_DATA_PATHS = [os.path.join(DATA_DIR, f'data_batch_{i}') for i in range(1, 6)]
TEST_DATA_PATH = 'data/cifar_test_nolabel.pkl'

# Directory paths
WEIGHTS_DIR = 'weights'
GRAPHS_DIR = 'graphs'
SCRIPTS_OUTPUT_DIR = 'scripts/outputs'
PREDICTIONS_DIR = 'predictions'

# Create directories
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(SCRIPTS_OUTPUT_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# Helper function to generate filenames with timestamp
def get_timestamped_filename(model, extension, directory=None):
    """Generate a filename using model name and current timestamp"""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = str(model).split('(')[0]  # Extract model class name
    filename = f"{model_name}_{timestamp}.{extension}"
    if directory:
        return os.path.join(directory, filename)
    return filename

# Dataset parameters
NUM_CLASSES = 10
IMAGE_SIZE = 32
CHANNELS = 3

# Augmentation transforms applied only to training data
# TRANSFORM = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
# ])

# Use AutoAugment for CIFAR10
TRANSFORM = transforms.Compose([
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10)
])

# DataLoader parameters
DATALOADER = {
    'batch_size': 512,
    'num_workers': 32,
    'pin_memory': True,
    'persistent_workers': True,
    'prefetch_factor': 32
}

# Optimizer parameters
OPTIMIZER = {
    'lr': 0.01,
    'weight_decay': 1e-4,
    'momentum': 0.9
}

# Scheduler parameters
SCHEDULER = {
    'mode': 'min',
    'factor': 0.1,
    'patience': 5,
    'min_lr': 1e-4
}

# Training parameters
TRAIN = {
    'epochs': 100,
    'early_stopping_patience': 10,
    'early_stopping_min_delta': 0.001,
    'mixup_alpha': 0.2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'no_augmentation_epochs': 10
}
