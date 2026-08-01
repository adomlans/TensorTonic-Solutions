def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    # Write code here
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    I = np.eye(X.shape[1])
    w = np.linalg.inv(X.T @ X + lam * I) @ X.T @ y
    return w.tolist()