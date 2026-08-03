import numpy as np

def r2_score(y_true, y_pred):
    """
    Compute R² (coefficient of determination) for 1D regression.
    Handle the constant-target edge case:
      - return 1.0 if predictions match exactly,
      - else 0.0.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    mean_y = np.mean(y_true)
    ss_tot = np.sum((y_true - mean_y) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return round(float(1.0 - ss_res / ss_tot), 4)
