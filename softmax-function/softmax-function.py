import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x = np.array(x, dtype=float)
    if x.ndim == 1:
        shifted = x - np.max(x)
        exps = np.exp(shifted)
        return np.round(exps / np.sum(exps), 4).tolist()
    elif x.ndim == 2:
        shifted = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(shifted)
        result = exps / np.sum(exps, axis=1, keepdims=True)
        return np.round(result, 4).tolist()
    pass