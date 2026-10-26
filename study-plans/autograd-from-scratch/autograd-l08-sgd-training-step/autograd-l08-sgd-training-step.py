import torch

def sgd_training_step(inputs, targets, weights, bias, learning_rate):
    """
    Returns: old and new loss, updated parameters, and current-step parameter gradients
    """
    learning_rate = float(learning_rate)
    x = inputs
    y = targets
    working_weights = weights.clone()
    working_bias = bias.clone()
    predictions = torch.tanh(x @ working_weights + working_bias)
    old_loss = ((predictions - y) ** 2).sum()
    preactivation_gradients = 2 * (predictions - y) * (1 - predictions * predictions)
    weight_gradients = x.transpose(0, 1) @ preactivation_gradients
    bias_gradient = preactivation_gradients.sum()
    updated_weights = working_weights - learning_rate * weight_gradients
    updated_bias = working_bias - learning_rate * bias_gradient
    new_predictions = torch.tanh(x @ updated_weights + updated_bias)
    new_loss = ((new_predictions - y) ** 2).sum()
    values = (old_loss, new_loss, updated_weights, updated_bias, weight_gradients, bias_gradient)
    return tuple((value.clone() for value in values))