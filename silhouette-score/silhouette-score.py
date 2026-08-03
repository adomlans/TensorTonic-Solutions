import numpy as np

def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Score for given points and cluster labels.
    X: np.ndarray of shape (n_samples, n_features)
    labels: np.ndarray of shape (n_samples,)
    Returns: float
    """
    X = np.asarray(X, float)
    labels = np.asarray(labels, int)
    n = len(X)
    dist = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    s = np.zeros(n)
    for i in range(n):
        same = labels == labels[i]
        a = np.mean(dist[i, same & (np.arange(n) != i)])
        b = np.min([
            np.mean(dist[i, labels == lab])
            for lab in np.unique(labels) if lab != labels[i]
        ])
        s[i] = (b - a) / max(a, b)
    return float(np.mean(s))
