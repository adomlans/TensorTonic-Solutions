def recursive_parameter_collection(weights, biases):
    """
    Returns: scalar parameter views in traversal order and the parameter count
    """
    parameters = []
    for weight, bias in zip(weights, biases):
        for neuron_index in range(weight.shape[0]):
            for input_index in range(weight.shape[1]):
                parameters.append(weight[neuron_index, input_index])
            parameters.append(bias[neuron_index])
    return (parameters, len(parameters))