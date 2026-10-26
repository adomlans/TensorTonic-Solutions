def value_multiplication_node(left, right, output_id):
    """
    Returns: a multiplication node that retains the two supplied leaf records as ordered parents
    """
    data = float(left['data']) * float(right['data'])
    return {'id': output_id, 'data': float(data), 'grad': 0.0, 'op': '*', 'parents': [left, right]}