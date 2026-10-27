import numpy as np

def select_puct_action(visits, total_values, priors, legal_mask, cpuct):
    """
    Returns: The selected action index and the complete PUCT score vector.
    """
    n = np.asarray(visits, dtype=np.int64)
    w = np.asarray(total_values, dtype=np.float64)
    p = np.asarray(priors, dtype=np.float64)
    legal = np.asarray(legal_mask, dtype=bool)

    q = np.zeros(n.size, dtype=np.float64)
    visited = n > 0
    q[visited] = w[visited] / n[visited]

    total_visits = np.sum(n, dtype=np.float64)
    exploration = cpuct * p * np.sqrt(total_visits) / (1.0 + n)
    scores = q + exploration
    scores[~legal] = -np.inf

    return int(np.argmax(scores)), scores