import torch
import torch.nn as nn
import os
import config as conf

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        os.makedirs(conf.WEIGHTS_DIR, exist_ok=True)
        self.weight_path = os.path.join(conf.WEIGHTS_DIR, f'{self}.pth')
        
    def save(self, path: str = None) -> None:
        """Save model weights to the specified path or default path.
        
        Args:
            path: Optional full path to save the model. If None, uses the default path.
        """
        path = path or self.weight_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"\nSaved model weights to {path}")
        
    def load(self, path: str = None) -> None:
        """Load model weights from the specified path or default path.
        
        Args:
            path: Optional full path to load the model from. If None, uses the default path.
        """
        path = path or self.weight_path
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, weights_only=True))
        else:
            print(f"No weights found at {path}. Using initialized weights.") 