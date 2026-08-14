import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class GRU:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_r = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_z = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_h = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.b_r = np.zeros(hidden_dim)
        self.b_z = np.zeros(hidden_dim)
        self.b_h = np.zeros(hidden_dim)
        self.W_y = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray) -> tuple:
        N, T, _ = X.shape
        h = np.zeros((N, self.hidden_dim))
        h_states = []
        for t in range(T):
            x_t = X[:, t, :]
            concat = np.concatenate([h, x_t], axis=-1)
            r_t = sigmoid(concat @ self.W_r.T + self.b_r)
            z_t = sigmoid(concat @ self.W_z.T + self.b_z)
            gated_h = r_t * h
            concat_h = np.concatenate([gated_h, x_t], axis=-1)
            h_tilde = np.tanh(concat_h @ self.W_h.T + self.b_h)
            h = z_t * h + (1 - z_t) * h_tilde
            h_states.append(h)
        all_h = np.stack(h_states, axis=1)
        y = all_h.reshape(-1, self.hidden_dim) @ self.W_y.T + self.b_y
        y = y.reshape(N, T, -1)
        return y, h
