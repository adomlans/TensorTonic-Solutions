def mean_rating_imputation(ratings_matrix, mode):
    """
    Fill missing ratings (zeros) with user or item means.
    """
    n_users = len(ratings_matrix)
    n_items = len(ratings_matrix[0])
    result = [row[:] for row in ratings_matrix]
    if mode == "user":
        for u in range(n_users):
            rated = [r for r in result[u] if r != 0]
            mean = sum(rated) / len(rated) if rated else 0.0
            for j in range(n_items):
                if result[u][j] == 0:
                    result[u][j] = mean
    else:
        item_means = []
        for j in range(n_items):
            rated = [result[u][j] for u in range(n_users) if result[u][j] != 0]
            item_means.append(sum(rated) / len(rated) if rated else 0.0)
        for u in range(n_users):
            for j in range(n_items):
                if result[u][j] == 0:
                    result[u][j] = item_means[j]
    return result
