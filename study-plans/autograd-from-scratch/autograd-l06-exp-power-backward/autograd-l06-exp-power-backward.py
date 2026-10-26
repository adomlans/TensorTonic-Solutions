import numpy as np

def exp_power_backward(x, exponent, upstream_gradient):
    """
    Returns: exponential and power outputs with their upstream-scaled gradients
    """
    x = float(x)
    exponent = float(exponent)
    upstream = float(upstream_gradient)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        exp_output = float(np.exp(x))
        exp_gradient = upstream * exp_output
        if exponent == 0.0:
            power_output = 1.0
            power_gradient = 0.0
        else:
            power_output = float(np.power(x, exponent))
            power_gradient = upstream * exponent * float(np.power(x, exponent - 1.0))
    values = (exp_output, exp_gradient, power_output, power_gradient)
    return values