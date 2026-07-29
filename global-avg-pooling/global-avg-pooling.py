import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    if x.ndim == 3:
        return np.mean(x, axis=(-2, -1))
    elif x.ndim == 4:
        return np.mean(x, axis=(-2, -1))
    else:
        raise ValueError("x must be (C,H,W) or (N,C,H,W)")
    pass