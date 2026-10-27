import numpy as np

def compute_advantage(states, rewards, V, gamma):
    rewards = np.asarray(rewards, dtype=float)
    values = np.asarray(V, dtype=float)
    returns = np.zeros(rewards.size, dtype=float)
    running = 0.0
    for index in range(rewards.size - 1, -1, -1):
        running = rewards[index] + gamma * running
        returns[index] = running
    advantages = returns - values[np.asarray(states, dtype=int)]
    return np.round(advantages, 4)