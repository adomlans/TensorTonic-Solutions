import numpy as np

def one_hot(y, num_classes=None):
    """
    Convert integer labels y ∈ {0,...,K-1} into one-hot matrix of shape (N, K).
    """
    y = list(y)
    if num_classes is None:
        num_classes = max(y) + 1
    result = []
    for label in y:
        row = [0] * num_classes
        row[label] = 1
        result.append(row)
    return result
