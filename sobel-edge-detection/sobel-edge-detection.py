def sobel_edges(image):
    """
    Apply the Sobel operator to detect edges.
    """
    # Write code here
    import math

def sobel_edges(image):
    """
    Apply the Sobel operator to detect edges.
    """
    h, w = len(image), len(image[0])
    padded = [[0]*(w+2) for _ in range(h+2)]
    for i in range(h):
        for j in range(w):
            padded[i+1][j+1] = image[i][j]
    Kx = [[-1,0,1],[-2,0,2],[-1,0,1]]
    Ky = [[-1,-2,-1],[0,0,0],[1,2,1]]
    out = []
    for i in range(h):
        row = []
        for j in range(w):
            gx = gy = 0.0
            for di in range(3):
                for dj in range(3):
                    v = padded[i+di][j+dj]
                    gx += v * Kx[di][dj]
                    gy += v * Ky[di][dj]
            row.append(math.sqrt(gx*gx + gy*gy))
        out.append(row)
    return out
