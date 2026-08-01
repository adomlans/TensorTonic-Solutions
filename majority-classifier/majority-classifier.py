import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    # Write code here
    y_train = np.asarray(y_train, dtype=int)
    X_test = np.asarray(X_test)
    if len(y_train) == 0:
        return np.array([])
    unique_classes, counts = np.unique(y_train, return_counts=True)
    majority_class = unique_classes[np.argmax(counts)]
    n_test = len(X_test) if X_test.ndim == 1 else X_test.shape[0]
    return np.full(n_test, majority_class, dtype=int)
    pass