def seasonal_average(series, period):
    """
    Compute the average value for each position in the seasonal cycle.
    """
    result = []
    for p in range(period):
        values = [series[i] for i in range(p, len(series), period)]
        result.append(sum(values) / len(values))
    return result
