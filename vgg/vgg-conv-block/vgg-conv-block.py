import numpy as np

def vgg_conv_block(x: np.ndarray, weights: list, biases: list) -> np.ndarray:
    out = x.copy()
    for W, b in zip(weights, biases):
        out = out @ W + b
        out = np.maximum(0, out)
    return out
