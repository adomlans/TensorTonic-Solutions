import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)
    N = len(y_true)
    p_true = y_pred[np.arange(N), y_true]
    return float(-np.mean(np.log(p_true)))
    pass