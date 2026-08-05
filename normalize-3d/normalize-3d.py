import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        norm = np.sqrt(np.sum(v**2))
        if norm < 1e-10:
            return v.copy()
        return v / norm
    else:
        norms = np.sqrt(np.sum(v**2, axis=1, keepdims=True))
        result = v.copy()
        mask = (norms.flatten() > 1e-10)
        result[mask] = v[mask] / norms[mask]
        return result
