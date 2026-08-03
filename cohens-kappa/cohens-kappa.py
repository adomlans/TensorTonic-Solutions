def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    """
    n = len(rater1)
    p_o = sum(1 for a, b in zip(rater1, rater2) if a == b) / n
    labels = set(rater1) | set(rater2)
    p_e = 0.0
    for label in labels:
        freq1 = sum(1 for x in rater1 if x == label) / n
        freq2 = sum(1 for x in rater2 if x == label) / n
        p_e += freq1 * freq2
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)
