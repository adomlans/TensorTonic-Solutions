import numpy as np

def visit_count_policy(visits, legal_mask, temperature, uniform_draw):
    """
    Returns: A tuple of the search policy array and selected action index.
    """
    counts = np.asarray(visits, dtype=np.int64)
    legal = np.asarray(legal_mask, dtype=bool)
    legal_indices = np.flatnonzero(legal)
    policy = np.zeros(counts.size, dtype=np.float64)

    if temperature == 0:
        action = int(legal_indices[np.argmax(counts[legal])])
        policy[action] = 1.0
        return policy, action

    positive = legal & (counts > 0)
    if not np.any(positive):
        policy[legal] = 1.0 / legal_indices.size
    else:
        log_weights = np.log(counts[positive].astype(np.float64)) / temperature
        log_weights = log_weights - log_weights.max()
        weights = np.exp(log_weights)
        policy[positive] = weights / weights.sum()

    cumulative = 0.0
    action = int(legal_indices[-1])
    for index in legal_indices:
        cumulative += float(policy[index])
        if uniform_draw < cumulative:
            action = int(index)
            break

    return policy, action