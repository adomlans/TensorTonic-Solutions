import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    # Write code here
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    intersection = np.sum(p.flatten() * y.flatten())
    return float(1.0 - (2.0 * intersection + eps) / (np.sum(p) + np.sum(y) + eps))
    pass