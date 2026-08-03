import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    """
    Compute confusion matrix with optional normalization.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    if num_classes is None:
        if len(y_true) == 0:
            num_classes = 0
        else:
            num_classes = max(int(y_true.max()), int(y_pred.max())) + 1
    K = num_classes
    if len(y_true) == 0:
        C = np.zeros((K, K), dtype=float)
    else:
        idx = y_true * K + y_pred
        C = np.bincount(idx, minlength=K*K).reshape(K, K).astype(float)
    if normalize == 'true':
        row_sums = C.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        C = C / row_sums
    elif normalize == 'pred':
        col_sums = C.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        C = C / col_sums
    elif normalize == 'all':
        total = C.sum()
        if total > 0:
            C = C / total
    return C
