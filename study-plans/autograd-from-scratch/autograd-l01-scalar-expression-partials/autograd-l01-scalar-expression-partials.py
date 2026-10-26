def scalar_expression_partials(a, b, c, h):
    """
    Returns: the expression value and its three numerical partial derivatives
    """
    values = (a, b, c, h)
    a_value, b_value, c_value, h_value = (float(value) for value in values)

    def expression(a_input, b_input, c_input):
        result = a_input * b_input + c_input
        return float(result)
    d = expression(a_value, b_value, c_value)
    partial_a = (expression(a_value + h_value, b_value, c_value) - d) / h_value
    partial_b = (expression(a_value, b_value + h_value, c_value) - d) / h_value
    partial_c = (expression(a_value, b_value, c_value + h_value) - d) / h_value
    return (d, float(partial_a), float(partial_b), float(partial_c))