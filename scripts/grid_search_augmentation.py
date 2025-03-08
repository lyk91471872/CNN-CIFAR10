#!/usr/bin/env python
"""
Grid search for optimal data augmentation (transforms) combinations.
This script trains models with different augmentation strategies using a fixed number of epochs
for fair comparison.
"""

import os
import sys
import csv
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
import itertools
from datetime import datetime
from tqdm import tqdm

# Add the project root to path to import project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config as conf
from dataset import create_dataset
from utils.pipeline import Pipeline
from utils.visualization import plot_training_history
from utils.session import SessionTracker

# Define the augmentation options to search through
AUGMENTATION_OPTIONS = {
    # Horizontal flip with different probabilities
    'hflip': [
        None,
        {'name': 'RandomHorizontalFlip', 'params': {'p': 0.5}},
    ],

    # ColorJitter with different parameters
    'color_jitter': [
        None,
        {'name': 'ColorJitter', 'params': {'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.1}},
        {'name': 'ColorJitter', 'params': {'brightness': 0.3, 'contrast': 0.3, 'saturation': 0.3, 'hue': 0.15}},
    ],

    # Affine transformations
    'affine': [
        None,
        {'name': 'RandomAffine', 'params': {'degrees': 10, 'translate': (0.1, 0.1), 'scale': (0.9, 1.1)}},
        {'name': 'RandomAffine', 'params': {'degrees': 15, 'translate': (0.15, 0.15), 'scale': (0.85, 1.15)}},
    ],

    # Random erasing (similar to Cutout)
    'erasing': [
        None,
        {'name': 'RandomErasing', 'params': {'p': 0.5, 'scale': (0.02, 0.2), 'ratio': (0.3, 3.3), 'value': 0}},
    ],

    # AutoAugment
    'auto_augment': [
        None,
        {'name': 'AutoAugment', 'params': {'policy': v2.AutoAugmentPolicy.CIFAR10}},
    ],
}

def create_transform_from_config(config_list):
    """Create a transform from a list of transform configurations."""
    transforms_list = []
    for c in config_list:
        if c is None:
            continue
        transforms_list.append(getattr(v2, c['name'])(**c['params']))
    transforms_list.append(v2.ToTensor())
    transforms_list.append(v2.Normalize(conf.CIFAR10_MEAN, conf.CIFAR10_STD))
    return v2.Compose(transforms_list)

def train_with_augmentation(transform, epochs, verbose=True):
    """Train a model with the given transform for a fixed number of epochs."""
    dataset = create_dataset(data_source=conf.TRAIN_DATA_PATHS, mode='training', transform=transform)
    model = conf.get_model()()
    pipeline = Pipeline(model, use_warmup=False, verbose=False)
    pipeline.early_stopping.patience = epochs   # Disable early stopping

    # Split into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], 
                                             generator=torch.Generator().manual_seed(42))

    # Create data loaders
    train_loader = DataLoader(train_dataset, shuffle=True, **conf.DATALOADER)
    val_loader = DataLoader(val_dataset, shuffle=False, **conf.DATALOADER)

    # Train for fixed number of epochs
    history = pipeline.train(train_loader, val_loader, should_save=False)

    # Get final metrics
    final_val_acc = history['val_accs'][-1] / 100.0  # Convert from percentage
    final_val_loss = history['val_losses'][-1]

    # Log results if verbose
    if verbose:
        print(f"Final validation accuracy: {final_val_acc*100:.2f}%")
        print(f"Final validation loss: {final_val_loss:.4f}")

    return {
        'val_acc': final_val_acc,
        'val_loss': final_val_loss,
        'history': history,
        'model': model
    }

def grid_search_augmentations(epochs=10, results_dir='scripts/outputs'):
    """Perform a grid search over all possible augmentation combinations."""
    print(f"\nStarting grid search for optimal data augmentation combinations")
    print(f"Training each combination for {epochs} epochs")

    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(results_dir, "augmentation_results.csv")

    # Initialize results
    results = []

    # Generate all possible combinations of augmentation types
    augmentation_keys = list(AUGMENTATION_OPTIONS.keys())

    # Create CSV file and write header
    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = ['name', 'val_acc', 'val_loss', 'epochs', 'transform_config']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    # Try all possible combinations of augmentation types
    for r in range(1, len(augmentation_keys) + 1):
        for keys_subset in itertools.combinations(augmentation_keys, r):
            print(f"\nTesting combinations of: {', '.join(keys_subset)}")

            # Generate all combinations of transforms for these keys
            transform_options = []
            for key in keys_subset:
                transform_options.append(AUGMENTATION_OPTIONS[key])

            # Generate all combinations for this set of keys
            combinations = list(itertools.product(*transform_options))

            for combination in tqdm(combinations, desc=f"Testing {'+'.join(keys_subset)}", leave=False):
                # Filter out None values
                transform_configs = [config for config in combination if config is not None]

                if not transform_configs:
                    continue  # Skip if no transforms selected

                # Create name for this combination
                combination_name = '+'.join([config['name'] for config in transform_configs])
                print(f"\nTesting: {combination_name}")

                # Create transform and train
                transform = create_transform_from_config(transform_configs)
                result = train_with_augmentation(transform, epochs)

                # Add metadata to result
                result['name'] = combination_name
                result['transform_configs'] = transform_configs
                result['epochs'] = epochs

                # Save model description, but not the actual model
                del result['model']

                # Append to results
                results.append(result)

                # Save to CSV
                with open(results_file, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    # Convert transform_configs to string for CSV
                    transform_config_str = str([{'name': c['name'], 'params': c['params']} for c in transform_configs])
                    writer.writerow({
                        'name': combination_name,
                        'val_acc': result['val_acc'],
                        'val_loss': result['val_loss'],
                        'epochs': epochs,
                        'transform_config': transform_config_str
                    })

    # Sort results by validation accuracy (descending)
    results.sort(key=lambda x: x['val_acc'], reverse=True)
    best_result = results[0]

    print("\n" + "="*80)
    print("Grid Search Results:")
    print("="*80)

    # Print top 5 results
    print("\nTop 5 Augmentation Combinations:")
    for i, result in enumerate(results[:5], 1):
        print(f"{i}. {result['name']}: {result['val_acc']*100:.2f}% accuracy")

    print("\nBest Augmentation Combination:")
    print(f"Name: {best_result['name']}")
    print(f"Validation Accuracy: {best_result['val_acc']*100:.2f}%")
    print(f"Transform Configuration:")
    for config in best_result['transform_configs']:
        print(f"  - {config['name']}: {config['params']}")

    print(f"\nDetailed results saved to: {results_file}")

    # Generate TRANSFORM configuration for the best combination
    best_transform_config = []
    for config in best_result['transform_configs']:
        transform_class = getattr(v2, config['name'])
        params_str = ', '.join(f"{k}={repr(v)}" for k, v in config['params'].items())
        transform_str = f"    v2.{config['name']}({params_str}),"
        best_transform_config.append(transform_str)

    print("\nRecommended TRANSFORM configuration for config.py:")
    print("TRANSFORM = v2.Compose([")
    for line in best_transform_config:
        print(line)
    print("    v2.ToTensor(),")
    print(f"    v2.Normalize({conf.CIFAR10_MEAN}, {conf.CIFAR10_STD})")
    print("])")

    return best_result, results

def main(epochs=10):
    """Main function to run the grid search."""
    try:
        best_result, all_results = grid_search_augmentations(epochs=epochs)
        return best_result, all_results
    except Exception as e:
        print(f"Error during grid search: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    # If run directly, use 10 epochs by default
    main(epochs=10) 
