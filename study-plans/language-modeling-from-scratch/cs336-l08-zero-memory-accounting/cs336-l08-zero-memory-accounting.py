def zero_memory_accounting(parameter_bytes, gradient_bytes, optimizer_bytes, world_size):
    """
    Returns: dictionary containing DDP and ZeRO byte totals
    """
    def shard(value):
        return (value + world_size - 1) // world_size

    return {
        "ddp": parameter_bytes + gradient_bytes + optimizer_bytes,
        "zero1": parameter_bytes + gradient_bytes + shard(optimizer_bytes),
        "zero2": parameter_bytes + shard(gradient_bytes + optimizer_bytes),
        "zero3": shard(parameter_bytes + gradient_bytes + optimizer_bytes),
    }