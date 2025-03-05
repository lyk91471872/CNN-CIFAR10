import torch
import numpy as np
from typing import Tuple

def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Returns mixed inputs, pairs of targets, and lambda.
    
    Args:
        x (torch.Tensor): Input data
        y (torch.Tensor): Target labels
        alpha (float): Alpha parameter for beta distribution
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]: Mixed inputs, target A, target B, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam 