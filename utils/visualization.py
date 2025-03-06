import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import os
from config import GRAPHS_DIR

def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None) -> None:
    """
    Plot training and validation metrics.
    
    Args:
        history (Dict[str, List[float]]): Dictionary containing training history
        save_path (str, optional): Path to save the plot. If None, saves to GRAPHS_DIR/training_history.png
    """
    if save_path is None:
        os.makedirs(GRAPHS_DIR, exist_ok=True)
        save_path = os.path.join(GRAPHS_DIR, 'training_history.png')
        
    plt.figure(figsize=(12, 4))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['val_losses'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accs'], label='Train Acc')
    plt.plot(history['val_accs'], label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")
    plt.close()

def plot_crossval_history(fold_results: List[Dict], save_path: Optional[str] = None) -> Dict[str, List[float]]:
    """
    Plot average training history across all cross-validation folds.
    
    Args:
        fold_results: List of dictionaries containing fold results with 'history' key
        save_path: Path to save the plot. If None, saves to GRAPHS_DIR/crossval_history.png
        
    Returns:
        Dict[str, List[float]]: The averaged history dictionary
    """
    if save_path is None:
        os.makedirs(GRAPHS_DIR, exist_ok=True)
        save_path = os.path.join(GRAPHS_DIR, 'crossval_history.png')
    
    # Create averaged history
    avg_history = {}
    
    # Check if all fold results have the same number of epochs
    min_epochs = min(len(result['history']['train_losses']) for result in fold_results)
    
    # Initialize with empty lists
    for key in fold_results[0]['history'].keys():
        avg_history[key] = []
    
    # Compute average for each epoch across all folds
    for epoch in range(min_epochs):
        for key in avg_history.keys():
            epoch_values = [result['history'][key][epoch] for result in fold_results]
            avg_history[key].append(np.mean(epoch_values))
    
    # Plot the averaged history
    plt.figure(figsize=(12, 4))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(avg_history['train_losses'], label='Train Loss')
    plt.plot(avg_history['val_losses'], label='Val Loss')
    plt.title('Average Loss Across Folds')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(avg_history['train_accs'], label='Train Acc')
    plt.plot(avg_history['val_accs'], label='Val Acc')
    plt.title('Average Accuracy Across Folds')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Cross-validation history plot saved to {save_path}")
    plt.close()
    
    return avg_history 