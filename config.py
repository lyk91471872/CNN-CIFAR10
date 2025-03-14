import os
import torch
from torchvision.transforms import v2
from pathlib import Path

# Base directory paths
ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / 'results'

# Directory paths
WEIGHTS_DIR = 'weights'
GRAPHS_DIR = 'graphs'
SCRIPTS_OUTPUT_DIR = 'scripts/outputs'
PREDICTIONS_DIR = 'predictions'
TRACKING_DIR = 'tracking'

# Dataset parameters
NUM_CLASSES = 10
IMAGE_SIZE = 32
CHANNELS = 3

# Normalization values for CIFAR-10
# Computed by update_normalization_values.py
CIFAR10_MEAN = (0.4914009, 0.48215896, 0.4465308)
CIFAR10_STD = (0.24703279, 0.24348423, 0.26158753)

# Base transforms (without augmentation)
BASE_TRANSFORM = v2.Compose([
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    v2.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])

'''
# Prevents adversarial attacks
# (No visible improvement on test acc)
BASE_TRANSFORM = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.uint8),  # Convert to uint8 for JPEG compression
    v2.JPEG(90),  # Apply JPEG compression
    v2.ToDtype(torch.float32, scale=True),  # Convert back to float32 for model input
    v2.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])
'''

# Augmentation transforms applied only to training data
TRANSFORM = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomCrop(32, padding=2),
    v2.RandomRotation(5),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
])

'''
# Progressive augmentations using RandAugment
# (No visible improvement on test acc)
# Applied on specific epochs, see utils/pipeline.py
TRANSFORM = v2.Compose([
    v2.RandAugment()
])
TRANSFORMF = v2.Compose([
    v2.RandAugment(10, 9)
])

TRANSFORMFF = v2.Compose([
    v2.RandAugment(20, 11)
])

TRANSFORMFFF = v2.Compose([
    v2.RandAugment(30, 13)
])

TRANSFORMFFFF = v2.Compose([
    v2.RandAugment(40, 15)
])
'''

'''
# Progressive augmentations using custom transforms
# (No visible improvement on test acc)
TRANSFORM = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    v2.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
])
TRANSFORMF = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.RandomPosterize(bits=2),
    v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
])

TRANSFORMFF = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.RandomPosterize(bits=4),
    v2.RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.8, 1.2)),
    v2.AutoAugment(policy=v2.AutoAugmentPolicy.CIFAR10)
])

TRANSFORMFFF = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
    v2.RandomPosterize(bits=4),
    v2.RandomAffine(degrees=15, translate=(0.3, 0.3), scale=(0.7, 1.3)),
    v2.AutoAugment(policy=v2.AutoAugmentPolicy.CIFAR10)
])

TRANSFORMFFF = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    v2.RandomPosterize(bits=4),
    v2.RandomAffine(degrees=20, translate=(0.4, 0.4), scale=(0.6, 1.4)),
    v2.AutoAugment(policy=v2.AutoAugmentPolicy.CIFAR10)
])
'''


# DataLoader parameters
DATALOADER = {
    'batch_size': 256,
    'num_workers': 32,
    'pin_memory': True,
    'persistent_workers': True,
    'prefetch_factor': 32
}

# Optimizer parameters
OPTIMIZER = {
    'lr': 0.1,
    'weight_decay': 2e-4,
    'momentum': 0.9
}

# Scheduler parameters
SCHEDULER = {
    'mode': 'min',
    'factor': 0.5,
    'patience': 5,
    'min_lr': 1e-5,
    'verbose': True
}

# Training parameters
TRAIN = {
    'epochs': 300,
    'early_stopping_patience': 30,
    'early_stopping_min_delta': 0.0005,
    'mixup_alpha': 1,   # alpha: draw lambda from a beta dist [0, 1]
    'cutmix_alpha': 1,  # alpha<0: convex, alpha=0: uniform, alpha>0: concave
    'cutmix_prob': 0.1,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'no_augmentation_epochs': 0,
    'min_save_epoch': 20,
    'warmup_epochs': 5
}

# Data paths
DATA_DIR = 'data/cifar-10-python/cifar-10-batches-py'
TRAIN_DATA_PATHS = [os.path.join(DATA_DIR, f'data_batch_{i}') for i in range(1, 6)]
TRAIN_DATA_PATHS.append(os.path.join(DATA_DIR, 'test_batch'))
TEST_DATA_PATH = 'data/cifar_test_nolabel.pkl'

# Model selection - using a function to avoid circular imports
def get_model():
    from models import CustomEfficientNetV2_B0, CustomResNet18, CustomResNet34, CustomResNet18X, PreActResNet18
    # Uncomment the line for the model you want to use
    # return CustomResNet18  # Standard ResNet18
    # return CustomResNet34  # Deeper ResNet34 (>5M, tried for distillation, not improving much)
    # return CustomResNet18X  # ResNet18 with custom channel size (using X value)
    return PreActResNet18  # Pre-activation ResNet18 (BN->ReLU->CONV order)
    # return CustomEfficientNetV2_B0  # EfficientNet

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Optimal channel size for CustomResNet18X
X = 42
