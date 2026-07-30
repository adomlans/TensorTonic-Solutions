def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    # Write code here
    dot = sum(a * b for a, b in zip(x1, x2))
    n1 = math.sqrt(sum(a * a for a in x1))
    n2 = math.sqrt(sum(b * b for b in x2))
    cos_sim = dot / (n1 * n2)
    if label == 1:
        return 1.0 - cos_sim
    else:
        return max(0.0, cos_sim - margin)

