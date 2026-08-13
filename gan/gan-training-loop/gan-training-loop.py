import numpy as np

def train_gan_step(real_data, fake_data, D_W):
    real_data = np.array(real_data, dtype=float)
    fake_data = np.array(fake_data, dtype=float)
    D_W = np.array(D_W, dtype=float)
    eps = 1e-8
    
    # Discriminator outputs
    real_probs = 1 / (1 + np.exp(-np.dot(real_data, D_W)))
    fake_probs = 1 / (1 + np.exp(-np.dot(fake_data, D_W)))
    
    real_probs = np.clip(real_probs, eps, 1 - eps)
    fake_probs = np.clip(fake_probs, eps, 1 - eps)
    
    d_loss = -np.mean(np.log(real_probs) + np.log(1 - fake_probs))
    g_loss = -np.mean(np.log(fake_probs))
    
    return {"d_loss": round(float(d_loss), 4), "g_loss": round(float(g_loss), 4)}
