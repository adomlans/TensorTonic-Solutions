import numpy as np

def vae_decoder(z, W_dec, b_dec):
    return np.dot(z, W_dec) + b_dec

