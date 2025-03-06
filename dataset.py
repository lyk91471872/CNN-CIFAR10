import numpy as np
import os
import pickle
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from config import TRANSFORM, BASE_TRANSFORM

def load_cifar_batch(file_path):
    """
    Loads a CIFAR-10 batch from a given file path.
    """
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    return batch

class CIFAR10Dataset(Dataset):
    """
    Flexible CIFAR-10 dataset that can handle both training and test data.
    
    Supports various modes:
    - Training mode: Returns (augmented_image, clean_image, label)
    - Test mode: Returns (transformed_image, index)
    - Raw mode: Returns (PIL_image, index)
    - Benchmark mode: Returns (transformed_image, label)
    """
    def __init__(self, data_source, mode='training', transform=None, return_labels=True, return_indices=False):
        """
        Initialize the dataset.
        
        Args:
            data_source: Either a list of file paths (training) or a single path (test).
            mode: One of 'training', 'test', 'raw', or 'benchmark'.
            transform: Transform to apply to the images.
            return_labels: Whether to return labels.
            return_indices: Whether to return indices.
        """
        self.mode = mode
        self.transform = transform
        self.return_labels = return_labels
        self.return_indices = return_indices
        
        # Set defaults based on mode
        if mode == 'training':
            self.aug_transform = transform or TRANSFORM
            self.base_transform = BASE_TRANSFORM
        elif mode == 'test':
            self.transform = transform or BASE_TRANSFORM
        elif mode == 'benchmark':
            self.transform = transform or BASE_TRANSFORM
        # For 'raw' mode, no transform is applied by default
        
        # Load the data
        self.data = []
        self.labels = []
        
        # Handle either a list of paths or a single path
        if isinstance(data_source, list):
            self._load_training_data(data_source)
        else:
            self._load_test_data(data_source)
            
    def _load_training_data(self, data_paths):
        """Load multiple training batches."""
        for path in data_paths:
            batch = load_cifar_batch(path)
            self.data.append(batch[b'data'])
            self.labels.extend(batch[b'labels'])
        # Reshape data to (N, 32, 32, 3)
        self.data = np.concatenate(self.data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        
    def _load_test_data(self, file_path):
        """Load a single test batch."""
        batch = load_cifar_batch(file_path)
        self.data = batch[b'data']  # Already in correct format
        if b'labels' in batch:
            self.labels = batch[b'labels']  # Use labels if they exist
        else:
            self.labels = []  # No labels for test data
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        pil_img = Image.fromarray(img)
        
        # Prepare return values based on mode
        result = []
        
        if self.mode == 'training':
            # Return (augmented_image, clean_image, label)
            clean_img = self.base_transform(pil_img)
            aug_img = self.base_transform(self.aug_transform(pil_img))
            result.append(aug_img)
            result.append(clean_img)
        elif self.mode == 'raw':
            # Return raw PIL image
            result.append(pil_img)
        else:  # 'test' or 'benchmark'
            # Return transformed image
            if self.transform:
                result.append(self.transform(pil_img))
            else:
                result.append(pil_img)
        
        # Add labels or indices as needed
        if self.return_labels and self.labels and idx < len(self.labels):
            result.append(self.labels[idx])
        
        if self.return_indices:
            result.append(idx)
            
        if len(result) == 1:
            return result[0]  # Return scalar if only one item
        return tuple(result)  # Otherwise return tuple

# Compatibility wrappers for existing code
def CIFAR10TestDataset(file_path, transform=BASE_TRANSFORM):
    """Compatibility wrapper for test dataset."""
    return CIFAR10Dataset(file_path, mode='test', transform=transform, 
                          return_labels=False, return_indices=True)

def CIFAR10BenchmarkDataset(data_paths, transform=BASE_TRANSFORM):
    """Compatibility wrapper for benchmark dataset."""
    return CIFAR10Dataset(data_paths, mode='benchmark', transform=transform)

def CIFAR10TestDatasetRaw(file_path):
    """Compatibility wrapper for raw test dataset."""
    return CIFAR10Dataset(file_path, mode='raw', return_labels=False, return_indices=True)
