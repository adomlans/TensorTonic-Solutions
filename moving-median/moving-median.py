def moving_median(values, window_size):
    """
    Compute the rolling median for each window position.
    """
    result = []
    for i in range(len(values) - window_size + 1):
        window = sorted(values[i:i + window_size])
        n = window_size
        if n % 2 == 1:
            result.append(float(window[n // 2]))
        else:
            result.append((window[n // 2 - 1] + window[n // 2]) / 2.0)
    return result
