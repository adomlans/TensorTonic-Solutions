import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    N, T, D_in = X.shape
    h_states = []
    h_curr = h_0
    for t in range(T):
        x_t = X[:, t, :]
        term_h = np.dot(h_curr, W_hh.T)
        term_x = np.dot(x_t, W_xh.T)
        h_curr = np.tanh(term_h + term_x + b_h)
        h_states.append(h_curr)
    hidden_states = np.stack(h_states, axis=1)
    return hidden_states, h_curr
