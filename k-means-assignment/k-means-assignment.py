def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    # Write code here
    result = []
    for p in points:
        best_j, best_d = 0, float('inf')
        for j, c in enumerate(centroids):
            d = sum((p[dim]-c[dim])**2 for dim in range(len(p)))
            if d < best_d: best_d = d; best_j = j
        result.append(best_j)
    return result