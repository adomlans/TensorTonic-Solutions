def kv_cache_capacity_planner(model_parameters, bytes_per_parameter, num_layers,
                              sequence_length, num_attention_heads, num_kv_heads,
                              head_dim, bytes_per_element, memory_per_gpu,
                              reserved_memory, memory_bandwidth):
    """
    Returns: dictionary containing cache capacity and bandwidth metrics
    """
    model_bytes = model_parameters * bytes_per_parameter
    kv_bytes_per_request = (
        2 * num_layers * sequence_length * num_kv_heads * head_dim * bytes_per_element
    )
    available = memory_per_gpu - reserved_memory - model_bytes
    max_batch_size = max(0, available // kv_bytes_per_request)
    bytes_per_step = model_bytes + max_batch_size * kv_bytes_per_request
    seconds_per_token = float(bytes_per_step / memory_bandwidth)
    tokens_per_second = (
        0.0 if max_batch_size == 0 else float(max_batch_size / seconds_per_token)
    )
    return {
        "model_bytes": model_bytes,
        "kv_bytes_per_request": kv_bytes_per_request,
        "max_batch_size": max_batch_size,
        "seconds_per_token": seconds_per_token,
        "tokens_per_second": tokens_per_second,
    }
