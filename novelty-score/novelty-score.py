import math

def novelty_score(recommendations, item_counts, n_users):
    """
    Compute the average novelty of a recommendation list.
    """
    total = 0.0
    for item in recommendations:
        pop = item_counts[item] / n_users
        if pop > 0:
            total += -math.log2(pop)
    return total / len(recommendations) if recommendations else 0.0
