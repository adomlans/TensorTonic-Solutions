import numpy as np

def initialize_mcts_edges(policy_logits, legal_mask):
    """
    Returns: A dictionary containing aligned N, W, Q, and P arrays.
    """
    logits = np.asarray(policy_logits, dtype=np.float64)
    legal = np.asarray(legal_mask, dtype=bool)

    priors = np.zeros(logits.size, dtype=np.float64)
    legal_logits = logits[legal]
    legal_logits = legal_logits - legal_logits.max()
    weights = np.exp(legal_logits)
    priors[legal] = weights / weights.sum()

    return {
        "N": np.zeros(logits.size, dtype=np.int64),
        "W": np.zeros(logits.size, dtype=np.float64),
        "Q": np.zeros(logits.size, dtype=np.float64),
        "P": priors,
    }