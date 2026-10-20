def collective_bandwidth(payload_bytes, world_size, duration_seconds, collective):
    """
    Returns: dictionary containing algorithm bytes and bandwidth
    """
    passes = 2 if collective == "all_reduce" else 1
    algorithm_bytes = float(payload_bytes * passes * (world_size - 1) / world_size)
    return {
        "algorithm_bytes": algorithm_bytes,
        "bandwidth_bytes_per_second": float(algorithm_bytes / duration_seconds),
    }