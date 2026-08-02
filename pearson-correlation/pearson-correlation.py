import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix from dataset X.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2:
        return None
    N = X.shape[0]
    mu = X.mean(axis=0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / (N - 1)
    std = np.sqrt(np.diag(cov))
    outer = np.outer(std, std)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(outer == 0, np.nan, cov / outer)
    np.fill_diagonal(corr, np.where(std == 0, np.nan, 1.0))
    return corr
