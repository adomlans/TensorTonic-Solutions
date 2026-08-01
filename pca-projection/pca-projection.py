import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    # Write code here
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    mean = X.mean(axis=0)
    X_c = X - mean
    cov = (X_c.T @ X_c) / (n - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    W = eigenvectors[:, :k]
    for j in range(k):
        max_idx = np.argmax(np.abs(W[:, j]))
        if W[max_idx, j] < 0:
            W[:, j] *= -1
    X_proj = X_c @ W
    return X_proj