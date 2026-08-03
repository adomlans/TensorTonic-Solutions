def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    classes = set(y_true) | set(y_pred)
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for c in classes:
        for yt, yp in zip(y_true, y_pred):
            if yp == c and yt == c:
                total_tp += 1
            elif yp == c and yt != c:
                total_fp += 1
            elif yp != c and yt == c:
                total_fn += 1
    denom = 2 * total_tp + total_fp + total_fn
    if denom == 0:
        return 0.0
    return round(2 * total_tp / denom, 4)
