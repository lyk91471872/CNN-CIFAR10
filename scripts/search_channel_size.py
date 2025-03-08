#!/usr/bin/env python
"""
Script to find the optimal channel size (x) for CustomResNet18X model.
The script searches for the largest x where the model has < 5M parameters.
"""

import torch
import sys
import os
import re
from torchsummary import summary

# Add the project root to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config as conf
from models.resnet import CustomResNet18X

def count_parameters(model):
    """Count the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def update_config_file(x_value):
    """Update the config.py file with the found X value."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.py')
    
    # Read the current config file
    with open(config_path, 'r') as f:
        config_content = f.read()
    
    # Check if X already exists in the config
    x_pattern = re.compile(r'^X\s*=\s*\d+', re.MULTILINE)
    if x_pattern.search(config_content):
        # Replace the existing X value
        new_config = x_pattern.sub(f'X = {x_value}', config_content)
    else:
        # Add X to the config file (after the CIFAR10_CLASSES list)
        new_config = config_content.rstrip() + f"\n\n# Optimal channel size for CustomResNet18X\nX = {x_value}\n"
    
    # Write the updated config
    with open(config_path, 'w') as f:
        f.write(new_config)
    
    print(f"Updated config.py with X = {x_value}")

def main():
    """
    Search for the largest value of x where CustomResNet18X has less than 5M parameters.
    Start from x=32 and increment until we find the threshold.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_params = 5_000_000  # 5 million parameter limit
    
    # Start the search from x=32
    x = 32
    found_x = x
    
    print(f"Searching for optimal channel size (x) with less than {max_params/1_000_000:.1f}M parameters")
    print(f"Using device: {device}")
    
    while True:
        # Create the model with current x
        model = CustomResNet18X(x=x)
        model.to(device)
        
        # Count parameters
        param_count = count_parameters(model)
        
        print(f"x = {x}: {param_count:,} parameters")
        
        # Check if we've exceeded the limit
        if param_count >= max_params:
            break
        
        # Save the current valid x and try the next one
        found_x = x
        x += 1
        
        # Set a reasonable upper limit to prevent infinite loops
        if x > 256:
            print("Reached upper limit of x=256, stopping search")
            break
    
    print(f"\nOptimal channel size: x = {found_x}")
    print(f"Parameters: {count_parameters(CustomResNet18X(x=found_x)):,}")
    
    # Update config.py with the found X value
    update_config_file(found_x)
    
    return found_x

if __name__ == "__main__":
    main() 