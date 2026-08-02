import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        return None
    N, D = X.shape
    if N < 2:
        return None
    mu = np.mean(X, axis=0)
    X_centered = X - mu
    cov_matrix = (1.0 / (N - 1)) * np.dot(X_centered.T, X_centered)
    return cov_matrix
