import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    g = np.asarray(g, dtype=float)
    if max_norm <= 0:
        return g.copy()
    norm = np.linalg.norm(g)
    if norm == 0 or norm <= max_norm:
        return g.copy()
    scale_factor = max_norm / norm
    return g * scale_factor
    # Write code here
    pass