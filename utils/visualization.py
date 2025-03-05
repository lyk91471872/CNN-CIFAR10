import matplotlib.pyplot as plt
from typing import Dict, List
import os
from config import GRAPHS_DIR

def plot_training_history(history: Dict[str, List[float]], save_path: str = None) -> None:
    """
    Plot training and validation metrics.
    
    Args:
        history (Dict[str, List[float]]): Dictionary containing training history
        save_path (str, optional): Path to save the plot. If None, saves to GRAPHS_DIR/training_history.png
    """
    if save_path is None:
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
    plt.close() 