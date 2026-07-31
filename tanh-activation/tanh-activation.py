import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    # Write code here
    x = np.array(x, dtype=float)
    return np.round((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)), 4).tolist()
    pass