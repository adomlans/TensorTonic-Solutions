import numpy as np

def mix_root_dirichlet_noise(priors, noise, legal_mask, epsilon):
    """
    Returns: A normalized root-prior array with noise mixed over legal actions.
    """
    p = np.asarray(priors, dtype=np.float64)
    eta = np.asarray(noise, dtype=np.float64)
    legal = np.asarray(legal_mask, dtype=bool)

    normalized_priors = np.zeros(p.size, dtype=np.float64)
    normalized_noise = np.zeros(p.size, dtype=np.float64)
    normalized_priors[legal] = p[legal] / p[legal].sum()
    normalized_noise[legal] = eta[legal] / eta[legal].sum()

    mixed = (1.0 - epsilon) * normalized_priors + epsilon * normalized_noise
    mixed[~legal] = 0.0
    return mixed