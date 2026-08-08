def discount_returns(rewards, gamma):
    """
    Compute the discounted return at every timestep.
    """
    n = len(rewards)
    G = [0.0] * n
    G[n - 1] = float(rewards[n - 1])
    for t in range(n - 2, -1, -1):
        G[t] = float(rewards[t]) + gamma * G[t + 1]
    return [round(g, 4) for g in G]
