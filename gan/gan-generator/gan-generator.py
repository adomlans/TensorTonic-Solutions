import numpy as np

def generator(z, W, b):
    z = np.array(z, dtype=float)
    W = np.array(W, dtype=float)
    b = np.array(b, dtype=float)
    out = np.tanh(np.dot(z, W) + b)
    return [[round(float(v), 4) for v in row] for row in out]
