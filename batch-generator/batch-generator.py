import numpy as np

def batch_generator(X, y, batch_size, rng=None, drop_last=False):
    """
    Randomly shuffle a dataset and yield mini-batches (X_batch, y_batch).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(X)
    indices = np.arange(n)
    if rng is not None:
        rng.shuffle(indices)
    else:
        np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        if drop_last and end - start < batch_size:
            break
        batch_indices = indices[start:end]
        yield X[batch_indices], y[batch_indices]
