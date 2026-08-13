import numpy as np

def vae_encoder(x, W_mu, b_mu, W_logvar, b_logvar):
    mu = np.dot(x, W_mu) + b_mu
    log_var = np.dot(x, W_logvar) + b_logvar
    return {"mu": mu, "log_var": log_var}
