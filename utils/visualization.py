import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix
from config import GRAPHS_DIR, CIFAR10_CLASSES

def plot_training_history(history: Dict[str, List[float]], model=None, epoch=None, accuracy=None, save_path: Optional[str] = None) -> None:
    """
    Plot training and validation metrics.
    
    Args:
        history (Dict[str, List[float]]): Dictionary containing training history
        model: Model object (for generating filename)
        epoch: Number of epochs trained (for filename)
        accuracy: Validation accuracy (for filename)
        save_path (str, optional): Path to save the plot. If None, uses session-based filename
    """
    from config import get_session_filename
    
    if save_path is None and model is not None:
        os.makedirs(GRAPHS_DIR, exist_ok=True)
        save_path = get_session_filename(
            model, 
            epoch=epoch, 
            accuracy=accuracy, 
            prefix="history", 
            extension="png", 
            directory=GRAPHS_DIR
        )
    elif save_path is None:
        os.makedirs(GRAPHS_DIR, exist_ok=True)
        save_path = os.path.join(GRAPHS_DIR, 'training_history.png')
        
    # Create the figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['val_losses'], label='Val Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracies
    ax2.plot(history['train_accs'], label='Train Acc')
    ax2.plot(history['val_accs'], label='Val Acc')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    
    # Add title showing model, epochs, and accuracy if available
    if model is not None and epoch is not None:
        model_name = model.__class__.__name__
        title = f"{model_name} - {epoch} epochs"
        if accuracy is not None:
            title += f" - {accuracy*100:.2f}% acc"
        fig.suptitle(title, fontsize=12)
        fig.subplots_adjust(top=0.85)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")
    plt.close()
    
    return save_path

def plot_confusion_matrix(y_true, y_pred, model=None, epoch=None, accuracy=None, save_path=None, normalize=True):
    """
    Plot confusion matrix for classification results.
    
    Args:
        y_true: Ground truth labels (list or numpy array)
        y_pred: Predicted labels (list or numpy array)
        model: Model object (for generating filename)
        epoch: Number of epochs trained (for filename)
        accuracy: Validation accuracy (for filename)
        save_path: Path to save the plot. If None, uses session-based filename
        normalize: Whether to normalize the confusion matrix (default: True)
    """
    from config import get_session_filename
    
    # Convert to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # Generate a filename if not provided
    if save_path is None and model is not None:
        os.makedirs(GRAPHS_DIR, exist_ok=True)
        save_path = get_session_filename(
            model, 
            epoch=epoch, 
            accuracy=accuracy, 
            prefix="confusion", 
            extension="png", 
            directory=GRAPHS_DIR
        )
    elif save_path is None:
        os.makedirs(GRAPHS_DIR, exist_ok=True)
        save_path = os.path.join(GRAPHS_DIR, 'confusion_matrix.png')
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                cmap='Blues', xticklabels=CIFAR10_CLASSES, yticklabels=CIFAR10_CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Add title showing model, epochs, and accuracy if available
    if model is not None:
        model_name = model.__class__.__name__
        title = f"Confusion Matrix - {model_name}"
        if epoch is not None:
            title += f" - {epoch} epochs"
        if accuracy is not None:
            title += f" - {accuracy*100:.2f}% acc"
        plt.title(title)
    else:
        plt.title("Confusion Matrix")
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix plot saved to {save_path}")
    plt.close()
    
    return save_path

def plot_crossval_history(fold_results: List[Dict], model=None, save_path: Optional[str] = None) -> Dict[str, List[float]]:
    """
    Plot average training history across all cross-validation folds.
    
    Args:
        fold_results: List of dictionaries containing fold results with 'history' key
        model: Model object (for generating filename)
        save_path: Path to save the plot. If None, generates a session-based path
        
    Returns:
        Dict[str, List[float]]: The averaged history dictionary
    """
    from config import get_session_filename
    
    if save_path is None and model is not None:
        os.makedirs(GRAPHS_DIR, exist_ok=True)
        
        # Calculate average validation accuracy
        best_val_accs = [result['best_val_acc'] for result in fold_results]
        avg_val_acc = np.mean(best_val_accs)
        
        save_path = get_session_filename(
            model, 
            accuracy=avg_val_acc, 
            prefix="crossval", 
            extension="png", 
            directory=GRAPHS_DIR
        )
    elif save_path is None:
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    ax1.plot(avg_history['train_losses'], label='Train Loss')
    ax1.plot(avg_history['val_losses'], label='Val Loss')
    ax1.set_title('Average Loss Across Folds')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracies
    ax2.plot(avg_history['train_accs'], label='Train Acc')
    ax2.plot(avg_history['val_accs'], label='Val Acc')
    ax2.set_title('Average Accuracy Across Folds')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    
    # Add title with average validation accuracy
    if model is not None:
        best_val_accs = [result['best_val_acc'] for result in fold_results]
        avg_val_acc = np.mean(best_val_accs)
        std_val_acc = np.std(best_val_accs)
        model_name = model.__class__.__name__
        title = f"{model_name} - Cross-validation ({len(fold_results)} folds)"
        title += f"\nAvg. Accuracy: {avg_val_acc*100:.2f}% ± {std_val_acc*100:.2f}%"
        fig.suptitle(title, fontsize=12)
        fig.subplots_adjust(top=0.85)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Cross-validation history plot saved to {save_path}")
    plt.close()
    
    return avg_history, save_path

def plot_crossval_confusion_matrices(fold_results, model=None, save_path=None):
    """
    Plot confusion matrices for each fold in cross-validation.
    
    Args:
        fold_results: List of dictionaries containing fold results
        model: Model object (for generating filename)
        save_path: Path to save the plot. If None, generates a session-based path
    """
    from config import get_session_filename
    
    if save_path is None and model is not None:
        os.makedirs(GRAPHS_DIR, exist_ok=True)
        
        # Calculate average validation accuracy
        best_val_accs = [result['best_val_acc'] for result in fold_results]
        avg_val_acc = np.mean(best_val_accs)
        
        save_path = get_session_filename(
            model, 
            accuracy=avg_val_acc, 
            prefix="crossval_confusion", 
            extension="png", 
            directory=GRAPHS_DIR
        )
    elif save_path is None:
        os.makedirs(GRAPHS_DIR, exist_ok=True)
        save_path = os.path.join(GRAPHS_DIR, 'crossval_confusion_matrices.png')
    
    # Calculate number of folds
    n_folds = len(fold_results)
    
    # Create figure with n_folds + 1 subplots (one for each fold plus one for average)
    fig_height = 4 * ((n_folds + 1) // 2 + (n_folds + 1) % 2)
    fig, axes = plt.subplots(((n_folds + 1) // 2 + (n_folds + 1) % 2), 2, 
                             figsize=(12, fig_height))
    axes = axes.flatten()
    
    # Plot confusion matrix for each fold
    combined_cm = None
    
    for i, result in enumerate(fold_results):
        if 'confusion_matrix' in result:
            cm = result['confusion_matrix']
        else:
            # Skip if no confusion matrix is available
            continue
            
        # Accumulate for average
        if combined_cm is None:
            combined_cm = cm
        else:
            combined_cm += cm
        
        # Normalize the confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot on the corresponding subplot
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=CIFAR10_CLASSES if i == n_folds-1 else [], 
                    yticklabels=CIFAR10_CLASSES if i % 2 == 0 else [],
                    ax=axes[i])
        axes[i].set_title(f"Fold {result['fold']} - Acc: {result['best_val_acc']*100:.2f}%")
        axes[i].set_xlabel('Predicted' if i >= n_folds - 2 else '')
        axes[i].set_ylabel('True' if i % 2 == 0 else '')
    
    # Plot average confusion matrix
    if combined_cm is not None:
        # Normalize the combined confusion matrix
        combined_cm_norm = combined_cm.astype('float') / combined_cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(combined_cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=CIFAR10_CLASSES, yticklabels=CIFAR10_CLASSES if n_folds % 2 == 0 else [],
                    ax=axes[n_folds])
        axes[n_folds].set_title(f"Average Across {n_folds} Folds")
        axes[n_folds].set_xlabel('Predicted')
        axes[n_folds].set_ylabel('True' if n_folds % 2 == 0 else '')
    
    # Hide any unused subplots
    for i in range(n_folds + 1, len(axes)):
        axes[i].set_visible(False)
    
    # Add overall title
    if model is not None:
        plt.suptitle(f"{model.__class__.__name__} - Cross-validation Confusion Matrices", 
                     fontsize=14, y=0.98)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.savefig(save_path)
    print(f"Cross-validation confusion matrices saved to {save_path}")
    plt.close()
    
    return save_path 