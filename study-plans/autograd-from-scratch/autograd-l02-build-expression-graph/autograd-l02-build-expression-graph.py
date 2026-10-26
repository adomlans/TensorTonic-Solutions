def build_expression_graph(leaves, operations):
    """
    Returns: node records in creation order and the final node ID
    """
    nodes = []
    by_id = {}
    for leaf in leaves:
        node_id = leaf['id']
        data = leaf['data']
        node = {'id': node_id, 'data': float(data), 'grad': 0.0, 'op': '', 'parents': []}
        nodes.append(node)
        by_id[node_id] = node
    for operation in operations:
        node_id = operation['id']
        op = operation['op']
        left_id = operation['left']
        right_id = operation['right']
        left_data = by_id[left_id]['data']
        right_data = by_id[right_id]['data']
        data = left_data + right_data if op == '+' else left_data * right_data
        node = {'id': node_id, 'data': float(data), 'grad': 0.0, 'op': op, 'parents': [left_id, right_id]}
        nodes.append(node)
        by_id[node_id] = node
    return (nodes, nodes[-1]['id'])