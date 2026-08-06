def robust_scaling(values):
    """
    Scale values using median and interquartile range.
    """
    n = len(values)
    s = sorted(values)
    if n % 2 == 1:
        median = s[n // 2]
    else:
        median = (s[n // 2 - 1] + s[n // 2]) / 2.0
    lower = s[:n // 2]
    upper = s[(n + 1) // 2:]
    def med(arr):
        m = len(arr)
        if m == 0:
            return 0.0
        if m % 2 == 1:
            return arr[m // 2]
        return (arr[m // 2 - 1] + arr[m // 2]) / 2.0
    q1 = med(lower)
    q3 = med(upper)
    iqr = q3 - q1
    if iqr == 0:
        return [(v - median) for v in values]
    return [(v - median) / iqr for v in values]
