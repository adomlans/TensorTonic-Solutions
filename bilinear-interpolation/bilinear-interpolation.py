def bilinear_resize(image, new_h, new_w):
    """
    Resize a 2D grid using bilinear interpolation.
    """
    h, w = len(image), len(image[0])
    out = []
    for i in range(new_h):
        row = []
        for j in range(new_w):
            sy = i * (h - 1) / (new_h - 1) if new_h > 1 else 0.0
            sx = j * (w - 1) / (new_w - 1) if new_w > 1 else 0.0
            y0 = int(sy)
            x0 = int(sx)
            y1 = min(y0 + 1, h - 1)
            x1 = min(x0 + 1, w - 1)
            dy = sy - y0
            dx = sx - x0
            v = image[y0][x0]*(1-dy)*(1-dx) + image[y1][x0]*dy*(1-dx) + image[y0][x1]*(1-dy)*dx + image[y1][x1]*dy*dx
            row.append(v)
        out.append(row)
    return out
