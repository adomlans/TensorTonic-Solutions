def popularity_ranking(items, min_votes, global_mean):
    """
    Compute the Bayesian weighted rating for each item.
    """
    result = []
    for avg_r, num_v in items:
        wr = (num_v / (num_v + min_votes)) * avg_r + (min_votes / (num_v + min_votes)) * global_mean
        result.append(wr)
    return result
