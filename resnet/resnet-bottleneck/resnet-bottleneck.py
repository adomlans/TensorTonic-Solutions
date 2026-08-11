import numpy as np

def bottleneck_block(x, W1, W2, W3, Ws):
    x = np.array(x, dtype=float)
    W1 = np.array(W1, dtype=float)
    W2 = np.array(W2, dtype=float)
    W3 = np.array(W3, dtype=float)
    if Ws is not None:
        Ws = np.array(Ws, dtype=float)
        identity = x @ Ws
    else:
        identity = x.copy()
    out = np.maximum(0, x @ W1)
    out = np.maximum(0, out @ W2)
    out = out @ W3
    result = np.maximum(0, out + identity)
    return [[round(float(v), 4) for v in row] for row in result]

