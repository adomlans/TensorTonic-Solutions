def histogram_equalize(image):
    """
    Apply histogram equalization to enhance image contrast.
    """
    h, w = len(image), len(image[0])
    total = h * w
    hist = [0] * 256
    for row in image:
        for v in row:
            hist[v] += 1
    cdf = [0] * 256
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]
    cdf_min = next(c for c in cdf if c > 0)
    denom = total - cdf_min
    mapping = [0] * 256
    if denom > 0:
        for i in range(256):
            if cdf[i] > 0:
                mapping[i] = round((cdf[i] - cdf_min) / denom * 255)
    return [[mapping[v] for v in row] for row in image]
