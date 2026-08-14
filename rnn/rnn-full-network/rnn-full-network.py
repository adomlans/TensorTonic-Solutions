import numpy as np

class VanillaRNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        self.W_xh = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / (2 * hidden_dim))
        self.W_hy = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_h = np.zeros(hidden_dim)
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray, h_0: np.ndarray = None) -> tuple:
        N, T, _ = X.shape
        if h_0 is None:
            h_0 = np.zeros((N, self.hidden_dim))
        h_curr = h_0
        h_states = []
        for t in range(T):
            x_t = X[:, t, :]
            h_curr = np.tanh(x_t @ self.W_xh.T + h_curr @ self.W_hh.T + self.b_h)
            h_states.append(h_curr)
        all_h = np.stack(h_states, axis=1)
        all_h_flat = all_h.reshape(-1, self.hidden_dim)
        y_flat = all_h_flat @ self.W_hy.T + self.b_y
        y_seq = y_flat.reshape(N, T, -1)
        return y_seq, h_curr
