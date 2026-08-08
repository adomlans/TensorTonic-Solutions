import numpy as np

def epsilon_greedy(q_values, epsilon, rng=None):
    """
    Returns: action index (int)
    """
    q_values = np.asarray(q_values)
    n_actions = len(q_values)
    if rng is not None:
        rand_val = rng.random()
    else:
        rand_val = 0.0
    if rand_val < epsilon:
        if rng is not None:
            action = int(rng.integers(0, n_actions))
        else:
            action = 0
        return action
    else:
        return int(np.argmax(q_values))
