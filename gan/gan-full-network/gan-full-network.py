import numpy as np

class GAN:
    def __init__(self, G_W, D_W):
        self.G_W = np.array(G_W, dtype=float)
        self.D_W = np.array(D_W, dtype=float)
    
    def generate(self, z):
        z = np.array(z, dtype=float)
        out = np.tanh(np.dot(z, self.G_W))
        return [[round(float(v), 4) for v in row] for row in out]
    
    def discriminate(self, x):
        x = np.array(x, dtype=float)
        logits = np.dot(x, self.D_W)
        probs = 1 / (1 + np.exp(-logits))
        return [[round(float(v), 4) for v in row] for row in probs]
    
    def train_step(self, real_data, z):
        eps = 1e-8
        fake_data = np.tanh(np.dot(np.array(z, dtype=float), self.G_W))
        real_probs = np.clip(1 / (1 + np.exp(-np.dot(np.array(real_data, dtype=float), self.D_W))), eps, 1-eps)
        fake_probs = np.clip(1 / (1 + np.exp(-np.dot(fake_data, self.D_W))), eps, 1-eps)
        d_loss = -np.mean(np.log(real_probs) + np.log(1 - fake_probs))
        g_loss = -np.mean(np.log(fake_probs))
        return {"d_loss": round(float(d_loss), 4), "g_loss": round(float(g_loss), 4)}
