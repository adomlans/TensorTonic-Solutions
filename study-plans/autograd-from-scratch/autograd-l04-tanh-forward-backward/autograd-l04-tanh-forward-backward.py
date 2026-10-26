import torch

def tanh_forward_backward(x, upstream_gradient):
    """
    Returns: tanh output and its upstream-scaled input gradient
    """
    output = torch.tanh(x)
    input_gradient = upstream_gradient * (1 - output.square())
    return (output, input_gradient)