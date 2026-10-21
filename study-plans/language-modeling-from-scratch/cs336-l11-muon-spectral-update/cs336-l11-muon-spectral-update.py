import torch

def muon_spectral_update(parameter, gradient, previous_momentum,
                         momentum_coefficient, learning_rate):
    """
    Returns: dictionary containing parameter, momentum, and spectral update
    """
    momentum = momentum_coefficient * previous_momentum + gradient
    u, _, vh = torch.linalg.svd(momentum.float(), full_matrices=False)
    update = (u @ vh).to(parameter.dtype)
    return {
        "new_parameter": parameter - learning_rate * update,
        "new_momentum": momentum,
        "orthogonalized_update": update,
    }