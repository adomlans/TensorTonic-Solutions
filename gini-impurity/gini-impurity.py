import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    # Write code here
    y_left = np.asarray(y_left, dtype=int)
    y_right = np.asarray(y_right, dtype=int)
    def gini_node(y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1.0 - np.sum(probs ** 2)
    n_total = len(y_left) + len(y_right)
    if n_total == 0:
        return 0.0
    return (len(y_left) / n_total) * gini_node(y_left) + (len(y_right) / n_total) * gini_node(y_right)
    pass