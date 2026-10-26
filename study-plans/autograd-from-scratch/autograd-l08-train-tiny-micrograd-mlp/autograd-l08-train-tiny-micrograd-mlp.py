import torch

def train_tiny_micrograd_mlp(inputs, targets, weights, biases, learning_rate, steps):
    """
    Returns: final predictions, final loss, trained parameters, and pre-update loss history
    """
    learning_rate = float(learning_rate)
    x = inputs
    y = targets
    trained_weights = [weight.clone() for weight in weights]
    trained_biases = [bias.clone() for bias in biases]
    loss_history = []

    def forward():
        activations = [x]
        current = x
        for weight, bias in zip(trained_weights, trained_biases):
            current = torch.tanh(current @ weight.transpose(0, 1) + bias)
            activations.append(current)
        predictions = current[:, 0]
        loss = ((predictions - y) ** 2).sum()
        return (predictions, loss, activations)
    for _ in range(steps):
        predictions, loss, activations = forward()
        loss_history.append(loss.clone())
        upstream = (2 * (predictions - y)).unsqueeze(1)
        weight_gradients = [None] * len(trained_weights)
        bias_gradients = [None] * len(trained_biases)
        for layer_index in range(len(trained_weights) - 1, -1, -1):
            output = activations[layer_index + 1]
            delta = upstream * (1 - output * output)
            weight_gradients[layer_index] = delta.transpose(0, 1) @ activations[layer_index]
            bias_gradients[layer_index] = delta.sum(dim=0)
            upstream = delta @ trained_weights[layer_index]
        trained_weights = [weight - learning_rate * gradient for weight, gradient in zip(trained_weights, weight_gradients)]
        trained_biases = [bias - learning_rate * gradient for bias, gradient in zip(trained_biases, bias_gradients)]
    final_predictions, final_loss, _ = forward()
    return (final_predictions, final_loss, trained_weights, trained_biases, loss_history)