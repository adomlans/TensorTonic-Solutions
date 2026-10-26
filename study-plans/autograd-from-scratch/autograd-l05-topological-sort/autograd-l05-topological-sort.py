def topological_sort(nodes, output_id):
    """
    Returns: reachable node IDs in deterministic topological order
    """
    parents = {}
    order = []
    for node in nodes:
        node_id = node['id']
        node_parents = node['parents']
        parents[node_id] = list(node_parents)
        order.append(node_id)
    reachable = set()
    stack = [output_id]
    while stack:
        node_id = stack.pop()
        if node_id in reachable:
            continue
        reachable.add(node_id)
        stack.extend(parents[node_id])
    position = {node_id: index for index, node_id in enumerate(order)}
    indegree = {node_id: len(parents[node_id]) for node_id in reachable}
    children = {node_id: [] for node_id in reachable}
    for node_id in reachable:
        for parent in parents[node_id]:
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
                insertion = 0
                while insertion < len(ready) and position[ready[insertion]] < position[child]:
                    insertion += 1
                ready.insert(insertion, child)
    return result