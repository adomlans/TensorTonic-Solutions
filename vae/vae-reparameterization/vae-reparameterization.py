import numpy as np

def reparameterize(mu, log_var, epsilon):
    std = np.exp(0.5 * log_var)
    return mu + std * epsilon

