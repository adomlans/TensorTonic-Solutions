import numpy as np

def conv_block(x, W1, W2, Ws):
    x = np.array(x, dtype=float)
    W1 = np.array(W1, dtype=float)
    W2 = np.array(W2, dtype=float)
    Ws = np.array(Ws, dtype=float)
    shortcut = x @ Ws
    out = np.maximum(0, x @ W1)
    out = out @ W2
    result = np.maximum(0, out + shortcut)
    return [[round(float(v), 4) for v in row] for row in result]
