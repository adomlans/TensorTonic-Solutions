import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    beta = np.asarray(beta, dtype=float)
    if x.ndim == 2:
        m = x.mean(axis=0, keepdims=True)
        v = x.var(axis=0, keepdims=True)
        xhat = (x - m) / np.sqrt(v + eps)
        return xhat * gamma[None, :] + beta[None, :]
    elif x.ndim == 4:
        m = x.mean(axis=(0,2,3), keepdims=True)
        v = x.var(axis=(0,2,3), keepdims=True)
        xhat = (x - m) / np.sqrt(v + eps)
        return xhat * gamma[None, :, None, None] + beta[None, :, None, None]
    else:
        raise ValueError("x must be (N,D) or (N,C,H,W)")
    pass