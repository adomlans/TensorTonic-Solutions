import numpy as np

def kfold_split(N, k, shuffle=True, rng=None):
    """
    Returns: list of length k with tuples (train_idx, val_idx)
    """
    indices = np.arange(N)
    if shuffle:
        if rng is not None:
            indices = rng.permutation(indices)
        else:
            np.random.shuffle(indices)
    fold_sizes = [N // k + (1 if i < N % k else 0) for i in range(k)]
    current = 0
    fold_indices = []
    for size in fold_sizes:
        fold_indices.append(indices[current:current + size])
        current += size
    result = []
    for i in range(k):
        val_idx = fold_indices[i]
        train_idx = np.concatenate([fold_indices[j] for j in range(k) if j != i])
        result.append((train_idx, val_idx))
    return result
