import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import from project root
import config as conf
from dataset import load_cifar_batch

def compute_normalization_values():
    """
    Compute mean and standard deviation for each channel in the CIFAR-10 training set.
    Returns:
        tuple: (means, stds) where each is a tuple of 3 values for RGB channels
    """
    print("Computing normalization values for CIFAR-10 training set...")
    
    # Load all training data
    all_data = []
    for path in conf.TRAIN_DATA_PATHS:
        batch = load_cifar_batch(path)
        data = batch[b'data']
        # Reshape from (N, 3072) to (N, 3, 32, 32)
        data = data.reshape(-1, 3, 32, 32)
        all_data.append(data)
    
    # Concatenate all batches
    all_data = np.concatenate(all_data, axis=0)
    
    # Convert to float and scale to [0, 1]
    all_data = all_data.astype(np.float32) / 255.0
    
    # Compute mean and std for each channel
    means = all_data.mean(axis=(0, 2, 3))
    stds = all_data.std(axis=(0, 2, 3))
    
    return tuple(means), tuple(stds)

def main():
    means, stds = compute_normalization_values()
    
    print("\nCIFAR-10 Normalization Values:")
    print(f"Means: {means}")
    print(f"Stds: {stds}")
    
    print("\nFor use in transforms.Normalize():")
    print(f"transforms.Normalize(mean={means}, std={stds})")
    
    # Save to a file for reference
    output_path = os.path.join(conf.SCRIPTS_OUTPUT_DIR, "normalization_values.txt")
    os.makedirs(conf.SCRIPTS_OUTPUT_DIR, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(f"CIFAR-10 Normalization Values:\n")
        f.write(f"Means: {means}\n")
        f.write(f"Stds: {stds}\n\n")
        f.write(f"For use in transforms.Normalize():\n")
        f.write(f"transforms.Normalize(mean={means}, std={stds})\n")
    
    print(f"\nValues saved to {output_path}")
    
    return means, stds

if __name__ == "__main__":
    main() 