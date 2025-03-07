# Standard library
import os
import json
import datetime
from typing import Dict, List, Optional

# Third-party
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Local application
from utils.session import SessionTracker, get_session_filename
import config as conf

# Base directory paths
ROOT_DIR = conf.ROOT_DIR
RESULTS_DIR = conf.RESULTS_DIR  # New central location for all results

# Directory paths
WEIGHTS_DIR = 'weights'
GRAPHS_DIR = 'graphs'
SCRIPTS_OUTPUT_DIR = 'scripts/outputs'
PREDICTIONS_DIR = 'predictions'
TRACKING_DIR = 'tracking'  # New directory for JSON tracking files

# Create directories
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(SCRIPTS_OUTPUT_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(TRACKING_DIR, exist_ok=True)

# Helper function to generate session-based filenames (replaces get_timestamped_filename)
def get_session_filename(model, epoch=None, accuracy=None, prefix=None, extension=None, directory=None):
    """
    Generate a consistent filename for all artifacts from a training/validation session.
    
    Args:
        model: The model object or name
        epoch: Number of epochs trained
        accuracy: Validation accuracy (0-1 range)
        prefix: Optional prefix to add to the filename
        extension: File extension (without dot)
        directory: Optional directory path
        
    Returns:
        The full path to the file
    """
    # Get model name
    if hasattr(model, '__class__'):
        model_name = model.__class__.__name__
    else:
        # Handle case where model is a string
        model_name = str(model).split('(')[0]
    
    # Format accuracy if provided
    acc_str = ""
    if accuracy is not None:
        acc_str = f"_A{int(100 * accuracy)}"
    
    # Format epoch if provided
    epoch_str = ""
    if epoch is not None:
        epoch_str = f"_E{epoch}"
    
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Build filename
    filename = f"{model_name}{epoch_str}{acc_str}_{timestamp}"
    
    # Add prefix if provided
    if prefix:
        filename = f"{prefix}_{filename}"
    
    # Add extension if provided
    if extension:
        filename = f"{filename}.{extension}"
    
    # Return full path if directory is provided
    if directory:
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, filename)
    
    return filename

class SessionTracker:
    """Class to handle tracking training/validation sessions with JSON."""
    
    def __init__(self, model, session_type="training"):
        """Initialize a new session tracker.
        
        Args:
            model: The model being trained/validated
            session_type: Either "training" or "crossval"
        """
        self.model_name = model.__class__.__name__
        self.session_type = session_type
        self.timestamp = datetime.datetime.now().isoformat()
        self.session_id = f"{self.model_name}_{session_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize data structure
        self.data = {
            "model_name": self.model_name,
            "session_type": session_type,
            "timestamp": self.timestamp,
            "config": {
                "optimizer": conf.OPTIMIZER,
                "scheduler": conf.SCHEDULER,
                "training": conf.TRAIN,
                "model_params": model.get_config() if hasattr(model, 'get_config') else {}
            },
            "metrics": {},
            "files": {}
        }
    
    def add_metrics(self, metrics):
        """Add metrics to the session data."""
        self.data["metrics"] = {**self.data.get("metrics", {}), **metrics}
        return self
    
    def add_file(self, file_type, file_path):
        """Add a file reference to the session data."""
        self.data["files"][file_type] = file_path
        return self
    
    def save(self):
        """Save the session data to a JSON file."""
        filename = f"{self.session_id}.json"
        filepath = os.path.join(TRACKING_DIR, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.data, f, indent=2)
        
        print(f"Session tracking data saved to {filepath}")
        return filepath
    
    @staticmethod
    def load(filepath):
        """Load session data from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Create a new SessionTracker and fill it with loaded data
        tracker = SessionTracker.__new__(SessionTracker)
        tracker.data = data
        tracker.model_name = data["model_name"]
        tracker.session_type = data["session_type"]
        tracker.timestamp = data["timestamp"]
        tracker.session_id = os.path.splitext(os.path.basename(filepath))[0]
        
        return tracker
    
    @staticmethod
    def list_sessions(model_name=None, session_type=None, limit=10):
        """List available sessions, optionally filtered by model name and/or type."""
        sessions = []
        
        for filename in os.listdir(TRACKING_DIR):
            if filename.endswith('.json'):
                filepath = os.path.join(TRACKING_DIR, filename)
                try:
                    tracker = SessionTracker.load(filepath)
                    
                    # Apply filters
                    if model_name and tracker.data["model_name"] != model_name:
                        continue
                    if session_type and tracker.data["session_type"] != session_type:
                        continue
                    
                    sessions.append(tracker)
                except Exception as e:
                    print(f"Error loading session from {filename}: {e}")
        
        # Sort by timestamp (newest first) and limit results
        sessions.sort(key=lambda x: x.data["timestamp"], reverse=True)
        return sessions[:limit]

# Dataset parameters
NUM_CLASSES = 10
IMAGE_SIZE = 32
CHANNELS = 3

# Normalization values for CIFAR-10
# These are placeholder values that will be updated by compute_normalization.py
CIFAR10_MEAN = (0.4914009, 0.48215896, 0.4465308)
CIFAR10_STD = (0.24703279, 0.24348423, 0.26158753)

# Base transforms (without augmentation)
BASE_TRANSFORM = conf.BASE_TRANSFORM

# Augmentation transforms applied only to training data
# TRANSFORM = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
# ])

# Use AutoAugment with additional basic augmentations
TRANSFORM = conf.TRANSFORM

# DataLoader parameters
DATALOADER = conf.DATALOADER

# Optimizer settings tuned to address underfitting:
# - Higher learning rate to escape local minima
# - Stronger regularization with higher weight decay
# - Learning rate warmup for stability
# - More gradual learning rate decay with higher patience
# - Enhanced data augmentation to improve generalization
OPTIMIZER = conf.OPTIMIZER

# Scheduler parameters
SCHEDULER = conf.SCHEDULER

# Training parameters
TRAIN = conf.TRAIN

# Data paths
DATA_DIR = 'data/cifar-10-python/cifar-10-batches-py'
TRAIN_DATA_PATHS = [os.path.join(DATA_DIR, f'data_batch_{i}') for i in range(1, 6)]
TEST_DATA_PATH = 'data/cifar_test_nolabel.pkl'

# Model selection - using a function to avoid circular imports
def get_model():
    """Get the model class to use for training.
    This function is used to avoid circular imports between config.py and models.
    """
    from models import CustomEfficientNetV2_B0, CustomResNet18
    return CustomEfficientNetV2_B0  # Change this to use a different model
