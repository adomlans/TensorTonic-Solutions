def local_vjp(operation, inputs, output, upstream_gradient):
    """
    Returns: one gradient contribution per input in input order
    """
    values = []
    for value in inputs:
        value = float(value)
        values.append(value)
    output = float(output)
    upstream_gradient = float(upstream_gradient)
    if operation == 'add':
        contributions = [upstream_gradient, upstream_gradient]
    elif operation == 'mul':
        contributions = [upstream_gradient * values[1], upstream_gradient * values[0]]
    else:
        contributions = [upstream_gradient * (1.0 - output ** 2)]
    return contributions