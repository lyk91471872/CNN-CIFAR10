import numpy as np
import os
import pickle
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from config import TRANSFORM
import random
import torch

def load_cifar_batch(file_path):
    """
    Loads a CIFAR-10 batch from a given file path.
    """
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    return batch

class CIFAR10Dataset(Dataset):
    """
    CIFAR-10 dataset with progressive learning support.
    
    Applies augmentations with increasing probability as training progresses.
    """
    def __init__(self, data_paths, transform=None, base_transform=None, augmentation_prob=0.0):
        self.data = []
        self.targets = []
        
        for path in data_paths:
            batch = load_cifar_batch(path)
            self.data.append(batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
            self.targets.extend(batch[b'labels'])
        
        self.data = np.vstack(self.data)
        self.transform = transform
        self.base_transform = base_transform
        self.augmentation_prob = augmentation_prob
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        
        # Convert to PIL Image
        pil_img = Image.fromarray(img)
        
        # Apply base transforms (always applied)
        if self.base_transform:
            img = self.base_transform(pil_img)
        else:
            # Default base transform if none provided
            base_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            img = base_transform(pil_img)
        
        # Apply augmentation with probability
        if self.transform and random.random() < self.augmentation_prob:
            # Convert back to PIL for augmentation if needed
            if not isinstance(img, Image.Image):
                # If img is a tensor, convert to PIL
                if isinstance(img, torch.Tensor):
                    img_np = img.mul(0.5).add(0.5).mul(255).byte().permute(1, 2, 0).numpy()
                    pil_img = Image.fromarray(img_np)
                    img = self.transform(pil_img)
                else:
                    img = self.transform(img)
            else:
                img = self.transform(img)
        
        return img, target
    
    def update_augmentation_prob(self, current_epoch, config):
        """Update the augmentation probability based on current epoch."""
        if not config['enabled']:
            self.augmentation_prob = 1.0
            return
            
        start_prob = config['start_prob']
        end_prob = config['end_prob']
        ramp_epochs = config['ramp_epochs']
        
        if current_epoch >= ramp_epochs:
            self.augmentation_prob = end_prob
        else:
            # Linear interpolation
            self.augmentation_prob = start_prob + (end_prob - start_prob) * (current_epoch / ramp_epochs)

class CIFAR10TestDataset(Dataset):
    """
    CIFAR-10 test dataset.
    
    Returns a tuple (transformed_image, index).
    """
    def __init__(self, file_path):
        batch = load_cifar_batch(file_path)
        self.data = batch[b'data']
        
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
