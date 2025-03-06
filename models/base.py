import torch
from torch import nn
import os
import config as conf
import sys

# Add project root to path to import utils.db
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.db import record_model_run, get_model_run_by_weights

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        os.makedirs(conf.WEIGHTS_DIR, exist_ok=True)
        # Default weight path will now use the timestamped filename
        self.weight_path = None  # Will be set on save()
        
    def save(self, path: str = None) -> None:
        """Save model weights to the specified path or default path.
        
        Args:
            path: Optional full path to save the model. If None, generates timestamped path.
        """
        if path is None:
            # Generate timestamped filename
            path = conf.get_timestamped_filename(self, 'pth', conf.WEIGHTS_DIR)
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"\nSaved model weights to {path}")
        
        # Set this as the current weight path
        self.weight_path = path
        
        # Record the model in the database
        try:
            # Create a config dictionary with relevant parameters
            config_dict = {
                'model_type': str(self),
                'optimizer': conf.OPTIMIZER,
                'scheduler': conf.SCHEDULER,
                'training': conf.TRAIN,
            }
            record_model_run(self, path, config_dict)
        except Exception as e:
            print(f"Warning: Failed to record model in database: {e}")
        
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
            self.load_state_dict(torch.load(path, weights_only=True))
            # Set this as the current weight path
            self.weight_path = path
            print(f"Loaded model weights from {path}")
        else:
            print(f"No weights found at {path}. Using initialized weights.") 