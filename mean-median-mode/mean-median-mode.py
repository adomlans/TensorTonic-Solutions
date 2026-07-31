import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    mean = float(np.mean(x))
    median = float(np.median(x))
    counts = Counter(x)
    max_count = max(counts.values())
    modes = [val for val, count in counts.items() if count == max_count]
    mode = float(min(modes))
    return mean, median, mode
    pass