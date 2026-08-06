def winsorize(values, lower_pct, upper_pct):
    """
    Clip values at the given percentile bounds.
    """
    s = sorted(values)
    n = len(s)
    def percentile(arr, p):
        if p <= 0:
            return arr[0]
        if p >= 100:
            return arr[-1]
        k = (len(arr) - 1) * p / 100.0
        f = int(k)
        c = f + 1
        if c >= len(arr):
            return arr[f]
        return arr[f] + (k - f) * (arr[c] - arr[f])
    lo = percentile(s, lower_pct)
    hi = percentile(s, upper_pct)
    return [max(lo, min(hi, v)) for v in values]
