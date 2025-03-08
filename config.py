import os
import torch
from torchvision.transforms import v2
from pathlib import Path

# Base directory paths
ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / 'results'  # New central location for all results

# Directory paths
WEIGHTS_DIR = 'weights'
GRAPHS_DIR = 'graphs'
SCRIPTS_OUTPUT_DIR = 'scripts/outputs'
PREDICTIONS_DIR = 'predictions'
TRACKING_DIR = 'tracking'  # New directory for JSON tracking files

# Dataset parameters
NUM_CLASSES = 10
IMAGE_SIZE = 32
CHANNELS = 3

# Normalization values for CIFAR-10
# These are placeholder values that will be updated by compute_normalization.py
CIFAR10_MEAN = (0.4914009, 0.48215896, 0.4465308)
CIFAR10_STD = (0.24703279, 0.24348423, 0.26158753)

# Base transforms (without augmentation)
BASE_TRANSFORM = v2.Compose([
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    v2.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])

# Augmentation transforms applied only to training data
TRANSFORM = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.2, hue=0.1),
    v2.RandomPosterize(bits=4),
    v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
])

'''
# Use AutoAugment with additional basic augmentations
TRANSFORM = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    v2.AutoAugment(policy=v2.AutoAugmentPolicy.CIFAR10)
])
'''

# DataLoader parameters
DATALOADER = {
    'batch_size': 512,
    'num_workers': 32,
    'pin_memory': True,
    'persistent_workers': True,
    'prefetch_factor': 32
}

# Optimizer settings tuned to address underfitting:
# - Higher learning rate to escape local minima
# - Stronger regularization with higher weight decay
# - Learning rate warmup for stability
# - More gradual learning rate decay with higher patience
# - Enhanced data augmentation to improve generalization
OPTIMIZER = {
    'lr': 0.05,  # Increased from 0.01 to help escape local minima
    'weight_decay': 5e-4,  # Increased regularization slightly
    'momentum': 0.9
}

# Scheduler parameters
SCHEDULER = {
    'mode': 'min',
    'factor': 0.5,
    'patience': 10,
    'min_lr': 1e-5,
    'verbose': True
}

# Training parameters
TRAIN = {
    'epochs': 300,
    'early_stopping_patience': 30,
    'early_stopping_min_delta': 0.0005,
    'mixup_alpha': 0.2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'no_augmentation_epochs': 0,
    'min_save_epoch': 20,
    'warmup_epochs': 10
}

# Data paths
DATA_DIR = 'data/cifar-10-python/cifar-10-batches-py'
TRAIN_DATA_PATHS = [os.path.join(DATA_DIR, f'data_batch_{i}') for i in range(1, 6)]
TEST_DATA_PATH = 'data/cifar_test_nolabel.pkl'

# Model selection - using a function to avoid circular imports
def get_model():
    """Get the model class to use for training.
    This function is used to avoid circular imports between config.py and models.
    """
    from models import CustomEfficientNetV2_B0, CustomResNet18, CustomResNet34, CustomResNet18X, PreActResNet18
    # Uncomment the line for the model you want to use
    # return CustomResNet18  # Standard ResNet18
    # return CustomResNet34  # Deeper ResNet34
    # return CustomResNet18X  # ResNet18 with custom channel size (using X value)
    return PreActResNet18  # Pre-activation ResNet18 (BN->ReLU->CONV order)
    # return CustomEfficientNetV2_B0  # EfficientNet

# ResNet18x channel size
X = 32

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Optimal channel size for CustomResNet18X
X = 32
