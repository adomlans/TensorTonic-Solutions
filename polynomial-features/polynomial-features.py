def polynomial_features(values, degree):
    """
    Generate polynomial features for each value up to the given degree.
    """
    return [[x ** p for p in range(degree + 1)] for x in values]
