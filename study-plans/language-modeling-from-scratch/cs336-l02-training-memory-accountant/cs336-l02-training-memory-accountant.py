def memory_accountant(param_shapes, param_bytes_per_element, grad_bytes_per_element, activation_shapes, activation_bytes_per_element, optimizer, optimizer_bytes_per_element):
    """
    Returns a dictionary of integer byte counts for parameters, gradients, activations, optimizer_state, and total.
    """

    def element_count(shapes):
        total = 0
        for shape in shapes:
            count = 1
            for dim in shape:
                count *= dim
            total += count
        return total
    param_elements = element_count(param_shapes)
    activation_elements = element_count(activation_shapes)
    parameters = param_elements * param_bytes_per_element
    gradients = param_elements * grad_bytes_per_element
    activations = activation_elements * activation_bytes_per_element
    state_multiplier = {'sgd': 0, 'adagrad': 1, 'adam': 2}[optimizer]
    optimizer_state = state_multiplier * param_elements * optimizer_bytes_per_element
    total = parameters + gradients + activations + optimizer_state
    return {'parameters': parameters, 'gradients': gradients, 'activations': activations, 'optimizer_state': optimizer_state, 'total': total}