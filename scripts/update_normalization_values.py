import os
import numpy as np
import re
import sys

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

def update_config_file(means, stds):
    """
    Update the config.py file with the computed normalization values.
    
    Args:
        means: Tuple of mean values for each channel
        stds: Tuple of standard deviation values for each channel
    """
    print("\nUpdating config.py with computed normalization values...")
    
    try:
        with open('config.py', 'r') as f:
            config_content = f.read()
        
        # Use regex to find and replace the normalization values
        mean_pattern = r'CIFAR10_MEAN\s*=\s*\([^)]*\)'
        std_pattern = r'CIFAR10_STD\s*=\s*\([^)]*\)'
        
        # Replace the values
        config_content = re.sub(mean_pattern, f'CIFAR10_MEAN = {means}', config_content)
        config_content = re.sub(std_pattern, f'CIFAR10_STD = {stds}', config_content)
        
        with open('config.py', 'w') as f:
            f.write(config_content)
        
        print("Config updated successfully.")
        print(f"New normalization values: mean={means}, std={stds}")
        print("Please restart for changes to take effect.")
        
        return True
    except Exception as e:
        print(f"Error updating config file: {e}")
        return False

def main():
    """
    Compute normalization values and update config.py.
    """
    try:
        means, stds = compute_normalization_values()
        
        print("\nCIFAR-10 Normalization Values:")
        print(f"Means: {means}")
        print(f"Stds: {stds}")
        
        print("\nUpdated normalization values in config.py.")
        print("\nFor use in transforms.Normalize():")
        print(f"v2.Normalize(mean={means}, std={stds})")
        
        # Update config.py
        update_config_file(means, stds)
        
        return means, stds
    except Exception as e:
        print(f"Error computing normalization values: {e}")
        return None, None

if __name__ == "__main__":
    main() 