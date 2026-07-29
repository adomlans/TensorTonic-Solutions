import numpy as np

def rnn_step_backward(dh, cache):
    """
    Returns:
        dx_t: gradient wrt input x_t      (shape: D,)
        dh_prev: gradient wrt previous h (shape: H,)
        dW: gradient wrt W               (shape: H x D)
        dU: gradient wrt U               (shape: H x H)
        db: gradient wrt bias            (shape: H,)
    """
    # Write code here
    x, h_prev, h_next, W, U, b = [np.asarray(c, dtype=float) for c in cache]
    dh = np.asarray(dh, dtype=float)
    dtanh = dh * (1.0 - h_next ** 2)
    db = dtanh.copy()
    dW = np.outer(dtanh, x)
    dU = np.outer(dtanh, h_prev)
    dx = W.T @ dtanh
    dh_prev = U.T @ dtanh
    return dx, dh_prev, dW, dU, db
    pass
