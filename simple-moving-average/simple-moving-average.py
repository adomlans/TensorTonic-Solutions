def simple_moving_average(values, window_size):
    """
    Compute the simple moving average of the given values.
    """
    result = []
    for i in range(len(values) - window_size + 1):
        window = values[i:i + window_size]
        result.append(sum(window) / window_size)
    return result
