def cache_aware_prefill_routing(request_token_counts, cached_prefix_lengths,
                                worker_ids, worker_capacities):
    """
    Returns: dictionary containing assignments and uncached worker loads
    """
    worker_count = len(worker_ids)
    assigned_tokens = [0] * worker_count
    uncached_loads = [0] * worker_count
    assignments = []
    for request_tokens, cache_row in zip(
        request_token_counts, cached_prefix_lengths
    ):
        candidates = []
        for worker_index in range(worker_count):
            if assigned_tokens[worker_index] + request_tokens > worker_capacities[worker_index]:
                continue
            uncached = request_tokens - cache_row[worker_index]
            projected_load = uncached_loads[worker_index] + uncached
            free_after = (
                worker_capacities[worker_index]
                - assigned_tokens[worker_index]
                - request_tokens
            )
            key = (
                -cache_row[worker_index],
                projected_load,
                -free_after,
                worker_ids[worker_index],
            )
            candidates.append((key, worker_index, uncached))
        if not candidates:
            assignments.append(-1)
            continue
        _, selected, uncached = min(candidates)
        assignments.append(worker_ids[selected])
        assigned_tokens[selected] += request_tokens
        uncached_loads[selected] += uncached
    return {
        "assignments": assignments,
        "worker_uncached_loads": uncached_loads,
    }