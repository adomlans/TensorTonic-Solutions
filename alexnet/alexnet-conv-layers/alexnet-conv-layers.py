import numpy as np

def alexnet_conv1(image: np.ndarray) -> np.ndarray:
    """AlexNet first conv layer: 11x11, stride 4, 96 filters (shape simulation)."""
    B = image.shape[0]
    # Output: (224 + 2*2 - 11) / 4 + 1 = 55
    return np.zeros((B, 55, 55, 96))
