def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    """
    H, W = len(image), len(image[0])
    result = []
    for i in range(H):
        row = []
        for j in range(W):
            r, g, b = image[i][j][0], image[i][j][1], image[i][j][2]
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            row.append(gray)
        result.append(row)
    return result
