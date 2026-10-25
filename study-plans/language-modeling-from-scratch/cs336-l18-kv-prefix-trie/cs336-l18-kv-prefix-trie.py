def kv_prefix_trie(cached_sequences, requests):
    """
    Returns: dictionary containing reusable-prefix match records
    """
    children = [{}]
    for sequence in cached_sequences:
        node = 0
        for token in sequence:
            if token not in children[node]:
                children[node][token] = len(children)
                children.append({})
            node = children[node][token]
    matches = []
    for sequence in requests:
        node = 0
        matched_length = 0
        for token in sequence:
            if token not in children[node]:
                break
            node = children[node][token]
            matched_length += 1
        matches.append({
            "matched_length": matched_length,
            "cache_node": node,
            "uncached_suffix": sequence[matched_length:],
        })
    return {"matches": matches}