import numpy as np

def discriminator(x, W):
    x = np.array(x, dtype=float)
    W = np.array(W, dtype=float)
    logits = np.dot(x, W)
    probs = 1 / (1 + np.exp(-logits))
    return [[round(float(v), 4) for v in row] for row in probs]
