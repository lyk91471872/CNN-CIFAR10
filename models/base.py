import os
import torch
import torch.nn as nn
from utils.session import get_session_filename
import config as conf

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        os.makedirs(conf.WEIGHTS_DIR, exist_ok=True)
        # Path to save/load weights
        self.weight_path = None
        # Current best validation accuracy
        self.best_val_accuracy = 0.0
        
    def save(self, epoch=None, accuracy=None, path=None) -> None:
        """Save model weights using the session-based naming scheme.
        
        Args:
            epoch: Number of epochs trained (optional)
            accuracy: Validation accuracy (optional)
            path: Optional full path to save the model. If None, generates a session-based path.
        """
        if path is None:
            # Generate session-based filename
            path = get_session_filename(
                self, 
                epoch=epoch, 
                accuracy=accuracy, 
                extension="pth", 
                directory=conf.WEIGHTS_DIR
            )
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        torch.save(self.state_dict(), path)
        print(f"\nSaved model weights to {path}")
        
        # Update current weight path
        self.weight_path = path
        
        # Record best validation accuracy if provided
        if accuracy is not None:
            self.best_val_accuracy = max(self.best_val_accuracy, accuracy)
        
        return path
        
    def load(self, path: str = None) -> None:
        """Load model weights from the specified path or default path.
        
        Args:
            path: Optional full path to load the model from. If None, uses the default path.
        """
        if path is None and self.weight_path is None:
            # No path specified and no saved path - try the latest model
            model_files = [f for f in os.listdir(conf.WEIGHTS_DIR) 
                          if f.startswith(str(self).split('(')[0]) and f.endswith('.pth')]
            if model_files:
                # Sort by modification time (newest first)
                model_files.sort(key=lambda x: os.path.getmtime(os.path.join(conf.WEIGHTS_DIR, x)), reverse=True)
                path = os.path.join(conf.WEIGHTS_DIR, model_files[0])
            else:
                print(f"No weights found for {str(self).split('(')[0]}. Using initialized weights.")
                return
        else:
            path = path or self.weight_path
            
        if os.path.exists(path):
            # Use weights_only=True to only load weights
            try:
                self.load_state_dict(torch.load(path, weights_only=True))
            except TypeError:
                # For older PyTorch versions that don't have weights_only parameter
                self.load_state_dict(torch.load(path))
                
            # Set this as the current weight path
            self.weight_path = path
            print(f"Loaded model weights from {path}")
        else:
            print(f"No weights found at {path}. Using initialized weights.")
            
    def get_config(self):
        """Get model configuration as a dictionary."""
        # Default implementation that derived classes can override
        return {
            "type": self.__class__.__name__,
            "params": str(self)
        } 