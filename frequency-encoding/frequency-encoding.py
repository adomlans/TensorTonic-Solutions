def frequency_encoding(values):
    """
    Replace each value with its frequency proportion.
    """
    n = len(values)
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return [counts[v] / n for v in values]
