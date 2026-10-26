import torch

def mlp_forward(inputs, weights, biases):
    """
    Returns: the final output tensor and all layer output tensors in forward order
    """
    activation = inputs
    layer_outputs = []
    for weight, bias in zip(weights, biases):
        activation = torch.tanh(weight @ activation + bias)
        layer_outputs.append(activation)
    return (activation, layer_outputs)