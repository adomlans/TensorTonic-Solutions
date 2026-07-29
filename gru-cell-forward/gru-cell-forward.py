import numpy as np

def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def _as2d(a, feat):
    """Convert 1D array to 2D and track if conversion happened"""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, feat), True
    return a, False

def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    Supports shapes (D,) & (H,) or (N,D) & (N,H).
    """
    # Write code here
    Wz, Uz, bz = params["Wz"], params["Uz"], params["bz"]
    Wr, Ur, br = params["Wr"], params["Ur"], params["br"]
    Wh, Uh, bh = params["Wh"], params["Uh"], params["bh"]
    x = np.asarray(x, dtype=float)
    h_prev = np.asarray(h_prev, dtype=float)
    D, H = Wz.shape
    single = (x.ndim == 1)
    if single:
        x = x.reshape(1, D)
        h_prev = h_prev.reshape(1, H)
    sigmoid = lambda a: np.where(a >= 0, 1/(1+np.exp(-a)), np.exp(a)/(1+np.exp(a)))
    z = sigmoid(x @ Wz + h_prev @ Uz + bz)
    r = sigmoid(x @ Wr + h_prev @ Ur + br)
    h_tilde = np.tanh(x @ Wh + (r * h_prev) @ Uh + bh)
    h = (1.0 - z) * h_prev + z * h_tilde
    if single:
        return h.reshape(H)
    return h
    pass