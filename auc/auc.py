import numpy as np
def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    if len(fpr) != len(tpr):
        raise ValueError("fpr and tpr must have same length")
    if len(fpr) < 2:
        raise ValueError("Need at least 2 points for AUC")
    if hasattr(np, 'trapezoid'):
        return float(np.trapezoid(tpr, fpr))
    else:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return float(np.trapz(tpr, fpr))

    pass