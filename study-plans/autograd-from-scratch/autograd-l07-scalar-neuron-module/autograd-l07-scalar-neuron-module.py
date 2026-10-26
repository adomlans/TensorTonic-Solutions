import torch

def scalar_neuron_module(inputs, weights, bias, nonlinear):
    """
    Returns: one scalar neuron output tensor
    """
    activation = (inputs * weights).sum() + bias
    output = torch.tanh(activation) if nonlinear else activation
    return output