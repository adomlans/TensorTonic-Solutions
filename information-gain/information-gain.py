import numpy as np

def _entropy(y):
    """
    Helper: Compute Shannon entropy (base 2) for labels y.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def information_gain(y, split_mask):
    """
    Compute Information Gain of a binary split on labels y.
    Use the _entropy() helper above.
    """
    # Write code here
    y = np.asarray(y)
    split_mask = np.asarray(split_mask)
    def entropy(arr):
        if arr.size == 0:
            return 0.0
        _, counts = np.unique(arr, return_counts=True)
        p = counts / counts.sum()
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())
    H_parent = entropy(y)
    left, right = y[split_mask], y[~split_mask]
    n = y.size
    if left.size == 0 or right.size == 0:
        return 0.0
    weighted = (left.size / n) * entropy(left) + (right.size / n) * entropy(right)
    return float(H_parent - weighted)
    pass
