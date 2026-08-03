def catalog_coverage(recommendations, n_items):
    """
    Compute the catalog coverage of a recommender system.
    """
    items = set()
    for rec in recommendations:
        items.update(rec)
    return len(items) / n_items if n_items > 0 else 0.0

    # Write code here