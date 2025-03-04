import torch
from enum import IntEnum

# Global flags
USE_AMP = True             # Enable automatic mixed precision
USE_MIXUP = True           # Enable mixup data augmentation
NUM_WORKERS = 64           # Number of DataLoader workers (adjust based on your CPU)
BATCH_SIZE = 256           # Batch size for training and evaluation
PREFETCH_FACTOR = 8        # DataLoader prefetch factor

# Optimizer and training hyperparameters
LEARNING_RATE = 0.1
WEIGHT_DECAY = 5e-4
PATIENCE = 10
MAX_EPOCHS = 100

# Model selection
class ModelTypeEnum(IntEnum):
    CUSTOM_RESNET18 = 0
    CUSTOM_EFFNETV2_B0 = 1
MODEL_TYPE = 1

# Data paths
DATA_DIR = "data/cifar-10-python/cifar-10-batches-py"
DATA_PATHS = [f"{DATA_DIR}/data_batch_{i}" for i in range(1, 6)]
TEST_FILE = "data/cifar_test_nolabel.pkl"

# Transform configurations (for reference in your transform definitions)
TRAIN_TRANSFORMS_CONFIG = {
    "RandomHorizontalFlip": True,
    "RandomRotation": 5,                # Rotation in degrees
    "RandomCrop_padding": 2,            # Padding for random crop
    "ColorJitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.1, "hue": 0.05},
    "ToTensor": True,
    "Normalize": {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
    # "RandAugment": {"num_ops": 2, "magnitude": 9},  # Uncomment if needed
    # "RandomErasing": {"p": 0.3, "scale": (0.02, 0.33), "ratio": (0.3, 3.3)}  # Uncomment if needed
}

TEST_TRANSFORMS_CONFIG = {
    "ToTensor": True,
    "Normalize": {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}
}

# Device configuration: Automatically detect GPU if available.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
