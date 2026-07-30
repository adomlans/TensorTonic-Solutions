import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    # Write code here
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    q_stable = q + eps
    kl = 0.0
    for i in range(len(p)):
        if p[i] > 0:
            kl += p[i] * np.log(p[i] / q_stable[i])
    return float(kl)
    pass