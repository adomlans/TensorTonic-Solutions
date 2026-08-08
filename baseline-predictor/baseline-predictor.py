def baseline_predict(ratings_matrix, target_pairs):
    """
    Compute baseline predictions using global mean and user/item biases.
    """
    all_ratings = [r for row in ratings_matrix for r in row if r != 0]
    mu = sum(all_ratings) / len(all_ratings)
    n_users = len(ratings_matrix)
    n_items = len(ratings_matrix[0])
    user_biases = []
    for u in range(n_users):
        u_ratings = [r for r in ratings_matrix[u] if r != 0]
        user_biases.append((sum(u_ratings) / len(u_ratings) - mu) if u_ratings else 0.0)
    item_biases = []
    for i in range(n_items):
        i_ratings = [ratings_matrix[u][i] for u in range(n_users) if ratings_matrix[u][i] != 0]
        item_biases.append((sum(i_ratings) / len(i_ratings) - mu) if i_ratings else 0.0)
    return [mu + user_biases[u] + item_biases[i] for u, i in target_pairs]
