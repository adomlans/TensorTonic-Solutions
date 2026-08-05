def conv2d(image, kernel, stride=1, padding=0):
    """
    Apply 2D convolution to a single-channel image.
    """
    h, w = len(image), len(image[0])
    kh, kw = len(kernel), len(kernel[0])
    if padding > 0:
        padded = [[0.0] * (w + 2 * padding) for _ in range(h + 2 * padding)]
        for i in range(h):
            for j in range(w):
                padded[i + padding][j + padding] = image[i][j]
    else:
        padded = [r[:] for r in image]
    ph, pw = len(padded), len(padded[0])
    oh = (ph - kh) // stride + 1
    ow = (pw - kw) // stride + 1
    out = []
    for i in range(oh):
        row = []
        for j in range(ow):
            v = 0.0
            for ki in range(kh):
                for kj in range(kw):
                    v += padded[i * stride + ki][j * stride + kj] * kernel[ki][kj]
            row.append(v)
        out.append(row)
    return out
