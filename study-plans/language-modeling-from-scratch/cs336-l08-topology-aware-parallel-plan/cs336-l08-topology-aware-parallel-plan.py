def topology_aware_parallel_plan(num_gpus, gpus_per_node, model_state_bytes,
                                 activation_bytes_per_sample, global_batch_size,
                                 sequence_length, memory_per_gpu,
                                 min_context_parallel):
    """
    Returns: best feasible plan dictionary or None
    """
    def ceil_div(value, divisor):
        return (value + divisor - 1) // divisor

    candidates = []
    for data_parallel in range(1, num_gpus + 1):
        if num_gpus % data_parallel or global_batch_size % data_parallel:
            continue
        after_data = num_gpus // data_parallel
        for tensor_parallel in range(1, after_data + 1):
            if after_data % tensor_parallel:
                continue
            after_tensor = after_data // tensor_parallel
            for pipeline_parallel in range(1, after_tensor + 1):
                if after_tensor % pipeline_parallel:
                    continue
                context_parallel = after_tensor // pipeline_parallel
                if context_parallel < min_context_parallel or sequence_length % context_parallel:
                    continue
                if tensor_parallel * context_parallel > gpus_per_node:
                    continue

                model_memory = ceil_div(model_state_bytes, tensor_parallel * pipeline_parallel)
                local_batch = global_batch_size // data_parallel
                activation_memory = ceil_div(
                    activation_bytes_per_sample * local_batch,
                    tensor_parallel * context_parallel,
                )
                total_memory = model_memory + activation_memory
                if total_memory > memory_per_gpu:
                    continue

                inter = float(
                    2 * model_memory * (data_parallel - 1) / data_parallel
                    + activation_memory * (pipeline_parallel - 1) / pipeline_parallel
                )
                intra = float(
                    2 * activation_memory
                    * ((tensor_parallel - 1) / tensor_parallel + (context_parallel - 1) / context_parallel)
                )
                score = (inter, intra, pipeline_parallel, -data_parallel, tensor_parallel, context_parallel)
                plan = {
                    "data_parallel": data_parallel,
                    "tensor_parallel": tensor_parallel,
                    "pipeline_parallel": pipeline_parallel,
                    "context_parallel": context_parallel,
                    "estimated_memory_bytes": total_memory,
                    "inter_node_bytes": inter,
                    "intra_node_bytes": intra,
                }
                candidates.append((score, plan))

    return min(candidates, key=lambda item: item[0])[1] if candidates else None
