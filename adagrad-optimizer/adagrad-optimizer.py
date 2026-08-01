import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """
    # Write code here
    w = np.array(w, dtype=float)
    g = np.array(g, dtype=float)
    G = np.array(G, dtype=float)
    G_new = G + g ** 2
    w_new = w - lr / np.sqrt(G_new + eps) * g
    return np.round(w_new, 6).tolist(), np.round(G_new, 6).tolist()
    pass