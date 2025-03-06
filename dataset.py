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
        
        # Configure transformations based on mode
        self._configure_transforms(transform)
        
        # Load the data
        self.data = []
        self.labels = []
        self._load_data(data_source)
            
    def _configure_transforms(self, transform):
        """Configure transforms based on the dataset mode."""
        if self.mode == 'training':
            self.aug_transform = transform or TRANSFORM
            self.base_transform = BASE_TRANSFORM
        elif self.mode in ('test', 'benchmark'):
            self.transform = transform or BASE_TRANSFORM
        # For 'raw' mode, no transform is needed
        
    def _load_data(self, data_source):
        """Load data from source, handling both training and test data."""
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
    
    def _get_pil_image(self, idx):
        """Convert raw data to PIL Image."""
        img = self.data[idx]
        return Image.fromarray(img)
    
    def _process_training_mode(self, pil_img):
        """Process image for training mode, returning (augmented, clean)."""
        clean_img = self.base_transform(pil_img)
        aug_img = self.base_transform(self.aug_transform(pil_img))
        return [aug_img, clean_img]
    
    def _process_test_mode(self, pil_img):
        """Process image for test or benchmark mode."""
        if self.transform:
            return [self.transform(pil_img)]
        return [pil_img]
    
    def _process_raw_mode(self, pil_img):
        """Return the raw PIL image without transformation."""
        return [pil_img]
    
    def _build_result(self, idx, processed_imgs):
        """Build result tuple with images, labels, and/or indices as needed."""
        result = processed_imgs.copy()
        
        # Add label if requested and available
        if self.return_labels and self.labels and idx < len(self.labels):
            result.append(self.labels[idx])
        
        # Add index if requested
        if self.return_indices:
            result.append(idx)
            
        # Return scalar for single items, tuple otherwise
        if len(result) == 1:
            return result[0]
        return tuple(result)
    
    def __getitem__(self, idx):
        """Get item based on the configured mode and return options."""
        pil_img = self._get_pil_image(idx)
        
        # Process image based on mode
        if self.mode == 'training':
            processed_imgs = self._process_training_mode(pil_img)
        elif self.mode == 'raw':
            processed_imgs = self._process_raw_mode(pil_img)
        else:  # 'test' or 'benchmark'
            processed_imgs = self._process_test_mode(pil_img)
        
        # Build and return final result
        return self._build_result(idx, processed_imgs)

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
