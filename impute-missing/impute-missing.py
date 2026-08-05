import numpy as np

def impute_missing(X, strategy="mean"):
    """
    Fill NaN values in each feature column using column mean or median.
    """
    X = np.asarray(X, dtype=float)
    X_imputed = X.copy()
    if X.ndim == 1:
        nan_mask = np.isnan(X)
        valid = X[~nan_mask]
        fill = np.mean(valid) if strategy == "mean" else np.median(valid)
        X_imputed[nan_mask] = fill if len(valid) > 0 else 0.0
    else:
        for col in range(X.shape[1]):
            column = X[:, col]
            nan_mask = np.isnan(column)
            valid = column[~nan_mask]
            fill = (np.mean(valid) if strategy == "mean" else np.median(valid)) if len(valid) > 0 else 0.0
            X_imputed[nan_mask, col] = fill
    return X_imputed
