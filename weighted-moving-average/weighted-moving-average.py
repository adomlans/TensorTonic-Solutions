def weighted_moving_average(values, weights):
    """
    Compute the weighted moving average using the given weights.
    """
    k = len(weights)
    w_sum = sum(weights)
    result = []
    for i in range(len(values) - k + 1):
        total = sum(weights[j] * values[i + j] for j in range(k))
        result.append(total / w_sum)
    return result
