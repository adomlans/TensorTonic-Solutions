def trace_reachable_graph(nodes, output_id):
    """
    Returns: reachable node IDs and parent-to-child edges in deterministic order
    """
    by_id = {}
    for node in nodes:
        node_id = node['id']
        by_id[node_id] = node
    reachable = set()
    stack = [output_id]
    while stack:
        node_id = stack.pop()
        if node_id in reachable:
            continue
        reachable.add(node_id)
        stack.extend(reversed(by_id[node_id]['parents']))
    reachable_ids = [node['id'] for node in nodes if node['id'] in reachable]
    edges = []
    for node in nodes:
        if node['id'] in reachable:
            for parent_id in node['parents']:
                edges.append([parent_id, node['id']])
    return (reachable_ids, edges)