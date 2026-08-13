import numpy as np

def discriminator_loss(real_probs, fake_probs):
    epsilon = 1e-8
    real_probs = np.clip(np.array(real_probs, dtype=float), epsilon, 1 - epsilon)
    fake_probs = np.clip(np.array(fake_probs, dtype=float), epsilon, 1 - epsilon)
    loss = -np.mean(np.log(real_probs) + np.log(1 - fake_probs))
    return round(float(loss), 4)

def generator_loss(fake_probs):
    epsilon = 1e-8
    fake_probs = np.clip(np.array(fake_probs, dtype=float), epsilon, 1 - epsilon)
    loss = -np.mean(np.log(fake_probs))
    return round(float(loss), 4)
