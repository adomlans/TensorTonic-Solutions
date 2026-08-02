import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return None
    det = np.linalg.det(A)
    if np.abs(det) < 1e-10:
        return None
    return np.linalg.inv(A)

