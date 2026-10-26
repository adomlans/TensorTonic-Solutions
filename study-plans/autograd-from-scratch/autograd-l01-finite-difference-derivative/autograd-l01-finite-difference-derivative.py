import numpy as np

def finite_difference_derivative(coefficients, x, h):
    """
    Returns: the polynomial value at x, the value at x plus h, and the forward-difference slope
    """
    values = np.asarray(coefficients, dtype=np.float64)
    x_value = float(x)
    h_value = float(h)

    def evaluate(point):
        result = np.float64(0.0)
        for coefficient in values[::-1]:
            result = result * point + coefficient
        return float(result)
    f_x = evaluate(x_value)
    f_x_plus_h = evaluate(x_value + h_value)
    slope = (f_x_plus_h - f_x) / h_value
    return (f_x, f_x_plus_h, float(slope))