def hit_rate_at_k(recommendations, ground_truth, k):
    """
    Compute the hit rate at K.
    """
    hits = 0
    for rec, gt in zip(recommendations, ground_truth):
        top_k = set(rec[:k])
        if top_k & set(gt):
            hits += 1
    return hits / len(recommendations) if recommendations else 0.0
