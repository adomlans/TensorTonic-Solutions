def morphological_op(image, kernel, operation):
    """
    Apply morphological erosion or dilation to a binary image.
    """
    h, w = len(image), len(image[0])
    kh, kw = len(kernel), len(kernel[0])
    ph, pw = kh // 2, kw // 2
    padded = [[0] * (w + 2 * pw) for _ in range(h + 2 * ph)]
    for i in range(h):
        for j in range(w):
            padded[i + ph][j + pw] = image[i][j]
    out = []
    for i in range(h):
        row = []
        for j in range(w):
            if operation == "erode":
                val = 1
                for di in range(kh):
                    for dj in range(kw):
                        if kernel[di][dj] == 1 and padded[i + di][j + dj] != 1:
                            val = 0; break
                    if val == 0: break
            else:
                val = 0
                for di in range(kh):
                    for dj in range(kw):
                        if kernel[di][dj] == 1 and padded[i + di][j + dj] == 1:
                            val = 1; break
                    if val == 1: break
            row.append(val)
        out.append(row)
    return out

    # Write code here