def image_histogram(image):
    """
    Compute the intensity histogram of a grayscale image.
    """
    hist = [0] * 256
    for row in image:
        for v in row:
            hist[v] += 1
    return hist
