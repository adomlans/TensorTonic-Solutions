import math

def ndcg(relevance_scores, k):
    """
    Compute NDCG@k.
    """
    scores = list(relevance_scores)[:k]
    dcg = sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(scores))
    ideal = sorted(relevance_scores, reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal))
    if idcg == 0:
        return 0.0
    return dcg / idcg
