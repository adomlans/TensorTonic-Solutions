def label_smoothing_loss(predictions, target, epsilon):
    """
    Compute cross-entropy loss with label smoothing.
    """
    # Write code here
    K = len(predictions)
    loss = 0.0
    for i in range(K):
        q = (1.0 - epsilon + epsilon / K) if i == target else (epsilon / K)
        loss -= q * math.log(predictions[i])
    return loss