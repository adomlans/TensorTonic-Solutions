import math

def gaussian_kernel(size, sigma):
    """
    Generate a normalized 2D Gaussian blur kernel.
    """
    center = size // 2
    kernel = []
    total = 0.0
    for i in range(size):
        row = []
        for j in range(size):
            x, y = j - center, i - center
            val = math.exp(-(x * x + y * y) / (2 * sigma * sigma))
            row.append(val)
            total += val
        kernel.append(row)
    for i in range(size):
        for j in range(size):
            kernel[i][j] /= total
    return kernel
