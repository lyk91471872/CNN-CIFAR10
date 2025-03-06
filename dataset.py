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
        
        # Apply base transform from config
        clean_img = BASE_TRANSFORM(pil_img)
        
        # Apply augmentation transform from config
        aug_img = TRANSFORM(pil_img)
        # Apply base transform after augmentation
        aug_img = BASE_TRANSFORM(aug_img)
            
        label = self.labels[idx]
        return aug_img, clean_img, label

class CIFAR10TestDataset(Dataset):
    """
    CIFAR-10 test dataset.
    
    Returns a tuple (transformed_image, index).
    """
    def __init__(self, file_path):
        batch = load_cifar_batch(file_path)
        self.data = batch[b'data'] # Do not reshape, it is already (N, 32, 32, 3)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        pil_img = Image.fromarray(img)
        
        # Apply base transform from config
        img = BASE_TRANSFORM(pil_img)
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
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        pil_img = Image.fromarray(img)
        
        # Apply base transform from config
        img = BASE_TRANSFORM(pil_img)
        label = self.labels[idx]
        return img, label
