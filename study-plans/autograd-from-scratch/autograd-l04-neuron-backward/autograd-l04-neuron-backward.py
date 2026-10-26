import torch

def neuron_backward(inputs, weights, bias, upstream_gradient):
    """
    Returns: output, input gradients, weight gradients, and bias gradient
    """
    x = inputs
    w = weights
    b = bias
    upstream = upstream_gradient
    output = torch.tanh(torch.sum(x * w) + b)
    local_gradient = upstream * (1 - output.square())
    input_gradients = local_gradient * w
    weight_gradients = local_gradient * x
    bias_gradient = local_gradient
    return (output, input_gradients, weight_gradients, bias_gradient)