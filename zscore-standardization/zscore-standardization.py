import numpy as np

def zscore_standardize(X, axis=0, eps=1e-12):
    """
    Standardize X: (X - mean)/std. If 2D and axis=0, per column.
    Return np.ndarray (float).
    """
    X = np.array(X, dtype=float)
    mu = X.mean(axis=axis, keepdims=True)
    sigma = X.std(axis=axis, keepdims=True, ddof=0)
    return ((X - mu) / (sigma + eps)).tolist()
