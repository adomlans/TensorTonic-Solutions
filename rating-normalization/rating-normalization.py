def rating_normalization(matrix):
    """
    Mean-center each user's ratings in the user-item matrix.
    """
    result = []
    for row in matrix:
        rated = [v for v in row if v != 0]
        if len(rated) == 0:
            result.append([0.0] * len(row))
            continue
        mean = sum(rated) / len(rated)
        result.append([v - mean if v != 0 else 0.0 for v in row])
    return result
