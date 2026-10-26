import torch

def dense_layer_forward(inputs, weight_matrix, biases, nonlinear):
    """
    Returns: one output tensor in neuron order
    """
    output = weight_matrix @ inputs + biases
    output = torch.tanh(output) if nonlinear else output
    return output