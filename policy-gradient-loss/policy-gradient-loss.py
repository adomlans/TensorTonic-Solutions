def policy_gradient_loss(log_probs, rewards, gamma):
    """
    Compute REINFORCE policy gradient loss with mean-return baseline.
    """
    T = len(rewards)
    G = [0.0] * T
    G[T - 1] = float(rewards[T - 1])
    for t in range(T - 2, -1, -1):
        G[t] = rewards[t] + gamma * G[t + 1]
    mean_G = sum(G) / T
    advantages = [g - mean_G for g in G]
    return -sum(lp * adv for lp, adv in zip(log_probs, advantages)) / T
