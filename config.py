import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
import os

# Create directories for saved files
os.makedirs('weights', exist_ok=True)
os.makedirs('graphs', exist_ok=True)

# Data paths
DATA_DIR = 'data/cifar-10-python/cifar-10-batches-py'
TRAIN_DATA_PATHS = [os.path.join(DATA_DIR, f'data_batch_{i}') for i in range(1, 6)]
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test_batch')

# Save paths
WEIGHTS_DIR = 'weights'
GRAPHS_DIR = 'graphs'

# Dataset parameters
NUM_CLASSES = 10
IMAGE_SIZE = 32
CHANNELS = 3

# Augmentation transforms applied only to training data
TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
])

# DataLoader parameters
DATALOADER = {
    'batch_size': 1024,
    'shuffle': True,
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
    'verbose': True,
    'min_lr': 1e-4
}

# Training parameters
TRAIN = {
    'epochs': 100,
    'early_stopping_patience': 10,
    'early_stopping_min_delta': 0.001,
    'mixup_alpha': 0.2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'use_cross_validation': True
} 