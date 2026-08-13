import numpy as np

def vae_loss(x, x_recon, mu, log_var):
    recon_loss = np.sum((x - x_recon)**2, axis=1).mean()
    kl_loss = -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var), axis=1).mean()
    total = recon_loss + kl_loss
    return {"total": float(total), "recon": float(recon_loss), "kl": float(kl_loss)}

