def bigram_probabilities(tokens):
    """
    Returns: (counts, probs)
      counts: dict mapping (w1, w2) -> integer count
      probs: dict mapping (w1, w2) -> float P(w2 | w1) with add-1 smoothing
    """
    if len(tokens) < 2:
        vocab = set(tokens)
        counts = {}
        probs = {}
        if len(tokens) == 1:
            probs[(tokens[0], tokens[0])] = 1.0
        return counts, probs
    vocab = set(tokens)
    V = len(vocab)
    counts = {}
    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i + 1]
        counts[(w1, w2)] = counts.get((w1, w2), 0) + 1
    context_totals = {}
    for w1 in vocab:
        context_totals[w1] = sum(counts.get((w1, w2), 0) for w2 in vocab)
    probs = {}
    for w1 in vocab:
        denom = context_totals[w1] + V
        for w2 in vocab:
            probs[(w1, w2)] = (counts.get((w1, w2), 0) + 1) / denom
    return counts, probs
