import numpy as np

def td_value_update(V, s, r, s_next, alpha, gamma):
    """
    Returns: updated value function V_new
    """
    V = list(V)
    td_target = r + gamma * V[s_next]
    V[s] = V[s] + alpha * (td_target - V[s])
    return [round(float(v), 4) for v in V]
