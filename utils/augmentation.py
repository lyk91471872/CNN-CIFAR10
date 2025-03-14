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

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.sqrt(1. - lam) * H

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def apply_cutmix(images, labels, alpha=1.0):
    '''
    images: Tensor of shape (batch_size, channels, H, W)
    labels: Tensor of shape (batch_size,) [LongTensor]
    alpha: CutMix hyperparameter (default=1.0)
    '''
    if alpha <= 0:
        return images, labels

    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    device = images.device

    # Permute batch
    rand_index = torch.randperm(batch_size).to(device)

    labels_a = labels
    labels_b = labels[rand_index]

    bbx1, bby1, bbx2, bby2 = (int(_) for _ in rand_bbox(images.size(), lam))
    
    # Apply CutMix by pasting the patch
    images[:, :, bby1:bby2, bbx1:bbx2] = images[rand_index, :, bby1:bby2, bbx1:bbx2]

    # Adjust lambda based on the actual area
    lam_adjusted = 1 - ((bbx2 - bbx1) * (bby2 - bby1)) / (images.size(-1) * images.size(-2))

    return images, labels_a, labels_b, lam_adjusted
