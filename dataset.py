import numpy as np
import os
import pickle
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from config import TRANSFORM

def load_cifar_batch(file_path):
    """
    Loads a CIFAR-10 batch from a given file path.
    """
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    return batch

class CIFAR10Dataset(Dataset):
    """
    CIFAR-10 training dataset.
    
    Returns a tuple (augmented_image, clean_image, label) where:
      - augmented_image: image processed by the provided transform.
      - clean_image: image processed by a base transform (ToTensor + Normalize).
      - label: corresponding label.
    """
    def __init__(self, data_paths):
        self.data = []
        self.labels = []
        for path in data_paths:
            batch = load_cifar_batch(path)
            self.data.append(batch[b'data'])
            self.labels.extend(batch[b'labels'])
        # Reshape data to (N, 32, 32, 3)
        self.data = np.concatenate(self.data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        pil_img = Image.fromarray(img)
        
        # Always apply base transforms
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        clean_img = base_transform(pil_img)
        
        # Apply augmentation transform from config
        aug_img = TRANSFORM(pil_img)
        # Apply base transforms after augmentation
        aug_img = base_transform(aug_img)
            
        label = self.labels[idx]
        return aug_img, clean_img, label

class CIFAR10TestDataset(Dataset):
    """
    CIFAR-10 test dataset.
    
    Returns a tuple (transformed_image, index).
    """
    def __init__(self, file_path):
        # Check if this is a pkl file (newer format) or a batch file (original format)
        if file_path.endswith('.pkl'):
            # New format - directly load numpy array
            with open(file_path, 'rb') as f:
                self.data = pickle.load(f)
            # Reshape to (N, 32, 32, 3) for PIL
            self.data = self.data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        else:
            # Original CIFAR batch format
            batch = load_cifar_batch(file_path)
            self.data = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        pil_img = Image.fromarray(img)
        
        # Always apply base transforms
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Always apply base transforms
        img = base_transform(pil_img)
        return img, idx

class CIFAR10BenchmarkDataset(Dataset):
    """
    Simplified CIFAR-10 dataset for benchmarking.
    Applies only base transforms (ToTensor + Normalize) without augmentation.
    """
    def __init__(self, data_paths):
        self.data = []
        self.labels = []
        for path in data_paths:
            batch = load_cifar_batch(path)
            self.data.append(batch[b'data'])
            self.labels.extend(batch[b'labels'])
        # Reshape data to (N, 32, 32, 3)
        self.data = np.concatenate(self.data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        
        # Base transform for normalization
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        pil_img = Image.fromarray(img)
        img = self.base_transform(pil_img)
        return img, self.labels[idx]
