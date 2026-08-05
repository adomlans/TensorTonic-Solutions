import numpy as np

def streaming_minmax_init(D):
    """
    Initialize state dict with min, max arrays of shape (D,).
    """
    return {'min': [float('inf')] * D, 'max': [float('-inf')] * D}

def streaming_minmax_update(state, X_batch, eps=1e-8):
    """
    Update state's min/max with X_batch, return normalized batch.
    """
    D = len(state['min'])
    for row in X_batch:
        for j in range(D):
            if row[j] < state['min'][j]:
                state['min'][j] = row[j]
            if row[j] > state['max'][j]:
                state['max'][j] = row[j]
    result = []
    for row in X_batch:
        norm_row = []
        for j in range(D):
            norm_row.append((row[j] - state['min'][j]) / (state['max'][j] - state['min'][j] + eps))
        result.append(norm_row)
    return result
