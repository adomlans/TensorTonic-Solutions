def min_max_scaling(data):
    """
    Scale each column of the data matrix to the [0, 1] range.
    """
    n_rows = len(data)
    n_cols = len(data[0])
    result = [[0.0] * n_cols for _ in range(n_rows)]
    for j in range(n_cols):
        col = [data[i][j] for i in range(n_rows)]
        col_min, col_max = min(col), max(col)
        rng = col_max - col_min
        for i in range(n_rows):
            result[i][j] = (data[i][j] - col_min) / rng if rng != 0 else 0.0
    return result

    # Write code here