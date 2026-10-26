import torch

def neuron_gradient_check(inputs, weights, bias, h):
    """
    Returns: analytic and numerical parameter gradients with their maximum error
    """
    h = float(h)
    x = inputs.to(dtype=torch.float64)
    w = weights.to(dtype=torch.float64)
    b = bias.to(dtype=torch.float64)
    base_output = torch.tanh(torch.sum(x * w) + b)
    local_gradient = 1 - base_output.square()
    analytic_weight_gradients = local_gradient * x
    numerical_weight_gradients = torch.empty_like(w)
    for index in range(w.numel()):
        perturbed_weights = w.clone()
        perturbed_weights[index] += h
        perturbed_output = torch.tanh(torch.sum(x * perturbed_weights) + b)
        numerical_weight_gradients[index] = (perturbed_output - base_output) / h
    analytic_bias_gradient = local_gradient
    numerical_bias_gradient = (torch.tanh(torch.sum(x * w) + b + h) - base_output) / h
    bias_error = torch.abs(analytic_bias_gradient - numerical_bias_gradient)
    if w.numel() == 0:
        max_error = bias_error
    else:
        weight_error = torch.max(torch.abs(analytic_weight_gradients - numerical_weight_gradients))
        max_error = torch.maximum(weight_error, bias_error)
    return (analytic_weight_gradients, numerical_weight_gradients, analytic_bias_gradient, numerical_bias_gradient, max_error)