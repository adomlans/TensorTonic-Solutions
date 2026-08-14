import numpy as np

def rnn_cell(x_t: np.ndarray, h_prev: np.ndarray,
             W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    term_h = np.dot(h_prev, W_hh.T)
    term_x = np.dot(x_t, W_xh.T)
    h_t = np.tanh(term_h + term_x + b_h)
    return h_t
