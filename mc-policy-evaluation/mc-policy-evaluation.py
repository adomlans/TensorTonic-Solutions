import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    """
    Returns: V (NumPy array of shape (n_states,))
    """
    returns_sum = [0.0] * n_states
    returns_count = [0] * n_states
    for episode in episodes:
        G = 0.0
        visit_return = {}
        for t in range(len(episode) - 1, -1, -1):
            state, reward = episode[t]
            G = reward + gamma * G
            visit_return[state] = G
        for state, ret in visit_return.items():
            returns_sum[state] += ret
            returns_count[state] += 1
    V = [0.0] * n_states
    for s in range(n_states):
        if returns_count[s] > 0:
            V[s] = round(returns_sum[s] / returns_count[s], 4)
    return V
