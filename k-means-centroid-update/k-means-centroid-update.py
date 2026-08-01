def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    # Write code here
    dim = len(points[0])
    sums = [[0.0] * dim for _ in range(k)]
    counts = [0] * k
    for i, p in enumerate(points):
        c = assignments[i]
        counts[c] += 1
        for d in range(dim):
            sums[c][d] += p[d]
    return [[sums[j][d] / counts[j] if counts[j] > 0 else 0.0
             for d in range(dim)] for j in range(k)]