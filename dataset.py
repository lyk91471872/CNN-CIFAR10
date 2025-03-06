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

class BaseCIFAR10Dataset(Dataset):
    """
    Base class for CIFAR-10 datasets.
    
    Handles common functionality like data loading and provides
    customization hooks through method overriding.
    """
    def __init__(self, data_source, transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_source: Either a list of file paths or a single file path.
            transform: Optional transform to apply to the images.
        """
        self.transform = transform
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
        self.labels = []  # No labels for test data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Base implementation - override in subclasses."""
        img = self.data[idx]
        pil_img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(pil_img)
        else:
            img = pil_img
            
        return img

class CIFAR10Dataset(BaseCIFAR10Dataset):
    """
    CIFAR-10 training dataset.
    
    Returns a tuple (augmented_image, clean_image, label) where:
      - augmented_image: image processed by the provided transform.
      - clean_image: image processed by a base transform (ToTensor + Normalize).
      - label: corresponding label.
    """
    def __init__(self, data_paths, transform=TRANSFORM, base_transform=BASE_TRANSFORM):
        """
        Initialize the training dataset.
        
        Args:
            data_paths: List of paths to CIFAR-10 batch files.
            transform: Transform for data augmentation.
            base_transform: Base transform for normalization.
        """
        super().__init__(data_paths)
        self.aug_transform = transform
        self.base_transform = base_transform
    
    def __getitem__(self, idx):
        img = self.data[idx]
        pil_img = Image.fromarray(img)
        
        # Apply base transform for clean image
        clean_img = self.base_transform(pil_img)
        
        # Apply augmentation transform
        aug_img = self.aug_transform(pil_img)
        # Apply base transform after augmentation
        aug_img = self.base_transform(aug_img)
            
        label = self.labels[idx]
        return aug_img, clean_img, label

class CIFAR10TestDataset(BaseCIFAR10Dataset):
    """
    CIFAR-10 test dataset.
    
    Returns a tuple (transformed_image, index).
    """
    def __init__(self, file_path, transform=BASE_TRANSFORM):
        """
        Initialize the test dataset.
        
        Args:
            file_path: Path to CIFAR-10 test batch file.
            transform: Transform to apply to images.
        """
        super().__init__(file_path, transform)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        pil_img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(pil_img)
        else:
            img = pil_img
            
        return img, idx

class CIFAR10BenchmarkDataset(BaseCIFAR10Dataset):
    """
    Simplified CIFAR-10 dataset for benchmarking.
    Applies only base transforms (ToTensor + Normalize) without augmentation.
    """
    def __init__(self, data_paths, transform=BASE_TRANSFORM):
        """
        Initialize the benchmark dataset.
        
        Args:
            data_paths: List of paths to CIFAR-10 batch files.
            transform: Transform to apply to images.
        """
        super().__init__(data_paths, transform)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        pil_img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(pil_img)
        else:
            img = pil_img
            
        label = self.labels[idx]
        return img, label

class CIFAR10TestDatasetRaw(BaseCIFAR10Dataset):
    """
    CIFAR-10 test dataset that returns raw images without normalization.
    
    Returns a tuple (raw_image, index) where raw_image is a PIL Image.
    """
    def __init__(self, file_path):
        """
        Initialize the raw test dataset.
        
        Args:
            file_path: Path to CIFAR-10 test batch file.
        """
        super().__init__(file_path, transform=None)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        # Convert to PIL Image directly without normalization
        pil_img = Image.fromarray(img)
        return pil_img, idx
