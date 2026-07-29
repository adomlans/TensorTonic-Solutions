import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    """
    scores: np.ndarray with shape (..., T, T)
    mask_value: float used to mask future positions (e.g., -1e9)
    Return: masked scores (same shape, dtype=float)
    """
    # Write code here
    s = np.asarray(scores, dtype=float)
    T = s.shape[-1]
    upper = np.triu(np.ones((T, T), dtype=bool), k=1)
    out = s.copy()
    out[..., upper] = mask_value
    return out
    pass