def target_encoding(categories, targets):
    """
    Replace each category with the mean target value for that category.
    """
    sums = {}
    counts = {}
    for cat, t in zip(categories, targets):
        sums[cat] = sums.get(cat, 0.0) + t
        counts[cat] = counts.get(cat, 0) + 1
    means = {cat: sums[cat] / counts[cat] for cat in sums}
    return [means[cat] for cat in categories]