import numpy as np

def _build_graph(leaves, operations):
    records = {}
    order = []
    for leaf in leaves:
        node_id = leaf['id']
        records[node_id] = {'kind': 'leaf', 'value': float(leaf['value']), 'parents': []}
        order.append(node_id)
    for operation in operations:
        node_id = operation['id']
        op = operation['op']
        parents = operation['parents']
        records[node_id] = {'kind': op, 'parents': list(parents)}
        order.append(node_id)
    return (records, order)

def _reachable(records, output_id):
    found = set()
    stack = [output_id]
    while stack:
        node_id = stack.pop()
        if node_id in found:
            continue
        found.add(node_id)
        stack.extend(records[node_id]['parents'])
    return found

def _stable_topological_order(records, order, output_id):
    reachable = _reachable(records, output_id)
    indegree = {node_id: sum((parent in reachable for parent in records[node_id]['parents'])) for node_id in reachable}
    children = {node_id: [] for node_id in reachable}
    position = {node_id: index for index, node_id in enumerate(order)}
    for node_id in reachable:
        for parent in records[node_id]['parents']:
            if parent in reachable:
                children[parent].append(node_id)
    for node_id in children:
        children[node_id].sort(key=position.get)
    ready = [node_id for node_id in order if node_id in reachable and indegree[node_id] == 0]
    result = []
    while ready:
        node_id = ready.pop(0)
        result.append(node_id)
        for child in children[node_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                insert_at = 0
                while insert_at < len(ready) and position[ready[insert_at]] < position[child]:
                    insert_at += 1
                ready.insert(insert_at, child)
    return result

def _forward_values(records, topological_order):
    values = {}
    for node_id in topological_order:
        record = records[node_id]
        kind = record['kind']
        if kind == 'leaf':
            values[node_id] = record['value']
        elif kind == 'add':
            values[node_id] = values[record['parents'][0]] + values[record['parents'][1]]
        elif kind == 'mul':
            values[node_id] = values[record['parents'][0]] * values[record['parents'][1]]
        else:
            values[node_id] = float(np.tanh(values[record['parents'][0]]))
    return values

def _reverse_gradients(records, topological_order, values, output_id):
    gradients = {node_id: 0.0 for node_id in topological_order}
    gradients[output_id] = 1.0
    for node_id in reversed(topological_order):
        record = records[node_id]
        upstream = gradients[node_id]
        if record['kind'] == 'leaf':
            continue
        parents = record['parents']
        if record['kind'] == 'add':
            contributions = (upstream, upstream)
        elif record['kind'] == 'mul':
            contributions = (upstream * values[parents[1]], upstream * values[parents[0]])
        else:
            contributions = (upstream * (1.0 - values[node_id] ** 2),)
        for parent, contribution in zip(parents, contributions):
            gradients[parent] += contribution
    return gradients

def autodiff_gradient_check(leaves, operations, output_id, h):
    """
    Returns: analytic gradients, numerical gradients, and their maximum error
    """
    h = float(h)
    records, order = _build_graph(leaves, operations)
    topological_order = _stable_topological_order(records, order, output_id)
    values = _forward_values(records, topological_order)
    gradients = _reverse_gradients(records, topological_order, values, output_id)
    reachable_leaf_ids = [leaf['id'] for leaf in leaves if leaf['id'] in gradients]
    analytic = {node_id: gradients[node_id] for node_id in reachable_leaf_ids}
    numerical = {}
    base_output = values[output_id]
    for target_id in reachable_leaf_ids:
        perturbed_leaves = [{'id': leaf['id'], 'value': leaf['value'] + h if leaf['id'] == target_id else leaf['value']} for leaf in leaves]
        perturbed_records, perturbed_order = _build_graph(perturbed_leaves, operations)
        perturbed_topological = _stable_topological_order(perturbed_records, perturbed_order, output_id)
        perturbed_values = _forward_values(perturbed_records, perturbed_topological)
        numerical[target_id] = (perturbed_values[output_id] - base_output) / h
    max_error = max((abs(analytic[node_id] - numerical[node_id]) for node_id in reachable_leaf_ids))
    return (analytic, numerical, max_error)