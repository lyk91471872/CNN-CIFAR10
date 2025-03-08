import numpy as np
import pickle
from PIL import Image
from torch.utils.data import Dataset
import config as conf

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
    def __init__(self, data_paths, transform=conf.TRANSFORM, base_transform=conf.BASE_TRANSFORM):
        self.data = []
        self.labels = []
        self.transform = transform
        self.base_transform = base_transform

        # Load all data batches
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

        # Apply base transform for clean image
        clean_img = self.base_transform(pil_img)

        # Apply augmentation transform
        aug_img = self.transform(pil_img)
        aug_img = self.base_transform(aug_img)

        label = self.labels[idx]
        return aug_img, clean_img, label

class CIFAR10TestDataset(Dataset):
    """
    CIFAR-10 test dataset.

    Returns a tuple (transformed_image, index).
    """
    def __init__(self, file_path, transform=conf.BASE_TRANSFORM):
        batch = load_cifar_batch(file_path)
        self.data = batch[b'data']  # Load raw data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        pil_img = Image.fromarray(img)

        if self.transform:
            img = self.transform(pil_img)

        return img, idx

class CIFAR10BenchmarkDataset(Dataset):
    """
    Simplified CIFAR-10 dataset for benchmarking.
    Applies only base transforms (ToTensor + Normalize) without augmentation.
    """
    def __init__(self, data_paths, transform=conf.BASE_TRANSFORM):
        self.data = []
        self.labels = []
        self.transform = transform

        # Load all data batches
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

        if self.transform:
            img = self.transform(pil_img)

        label = self.labels[idx]
        return img, label

class CIFAR10TestDatasetRaw(Dataset):
    """
    CIFAR-10 test dataset that returns raw images without normalization.

    Returns a tuple (raw_image, index) where raw_image is a PIL Image.
    """
    def __init__(self, file_path):
        batch = load_cifar_batch(file_path)
        self.data = batch[b'data']  # Load raw data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        pil_img = Image.fromarray(img)
        return pil_img, idx

# For backward compatibility with existing code
def create_dataset(data_source, mode='training', transform=None, raw=False):
    """Simple factory function for backward compatibility."""
    if mode == 'training':
        return CIFAR10Dataset(data_paths=data_source, transform=transform or conf.TRANSFORM)
    elif mode == 'test':
        if raw:
            return CIFAR10TestDatasetRaw(file_path=data_source)
        return CIFAR10TestDataset(file_path=data_source, transform=transform or conf.BASE_TRANSFORM)
    elif mode == 'benchmark':
        return CIFAR10BenchmarkDataset(data_paths=data_source, transform=transform or conf.BASE_TRANSFORM)
    else:
        raise ValueError(f"Unknown mode: {mode}")
