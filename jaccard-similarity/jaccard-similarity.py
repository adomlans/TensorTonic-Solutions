def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    a = set(set_a)
    b = set(set_b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union
