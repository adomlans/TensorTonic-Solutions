def sarsa_update(q_table, state, action, reward, next_state, next_action, alpha, gamma):
    """
    Perform one SARSA update and return the updated Q-table.
    """
    Q = [list(row) for row in q_table]
    Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
    Q = [[round(float(v), 4) for v in row] for row in Q]
    return Q
