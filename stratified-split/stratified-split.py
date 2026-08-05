import numpy as np

def stratified_split(X, y, test_size, rng=None):
    """
    Split features X and labels y into train/test while preserving class proportions.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    classes = sorted(set(y.tolist()))
    train_indices, test_indices = [], []
    for c in classes:
        c_idx = np.where(y == c)[0].copy()
        if rng is not None:
            c_idx = rng.permutation(c_idx)
        n_test = int(round(len(c_idx) * test_size))
        if n_test >= len(c_idx) and len(c_idx) > 1:
            n_test = len(c_idx) - 1
        test_indices.extend(c_idx[:n_test].tolist())
        train_indices.extend(c_idx[n_test:].tolist())
    train_idx = np.array(sorted(train_indices), dtype=int)
    test_idx = np.array(sorted(test_indices), dtype=int)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
