import numpy as np

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 samples for t-test")
    x_bar = np.mean(x)
    s = np.sqrt(np.sum((x - x_bar) ** 2) / (n - 1))
    se = s / np.sqrt(n)
    t_stat = (x_bar - mu0) / se
    return float(t_stat)
    pass