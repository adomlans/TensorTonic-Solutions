def value_addition_node(left, right, output_id):
    """
    Returns: an addition node that retains the two supplied leaf records as ordered parents
    """
    data = float(left['data']) + float(right['data'])
    return {'id': output_id, 'data': float(data), 'grad': 0.0, 'op': '+', 'parents': [left, right]}