def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """
    # Write code here
    n = len(predictions)
    total = 0.0
    for i in range(n):
        p_t = predictions[i] if targets[i] == 1 else 1.0 - predictions[i]
        total -= alpha * ((1.0 - p_t) ** gamma) * math.log(p_t)
    return total / n