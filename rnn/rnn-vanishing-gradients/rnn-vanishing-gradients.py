import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    s = np.linalg.norm(W_hh, ord=2)
    norms = []
    current_grad = 1.0
    for t in range(T):
        norms.append(current_grad)
        current_grad *= s
    return norms
