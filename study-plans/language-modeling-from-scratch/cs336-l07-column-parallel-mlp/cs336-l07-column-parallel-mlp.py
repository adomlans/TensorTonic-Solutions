import torch

def column_parallel_mlp(x, weight_shards, bias_shards=None):
    """
    Returns: dictionary containing local and full activations
    """
    local_activations = []
    for rank, weight in enumerate(weight_shards):
        local = x @ weight
        if bias_shards is not None:
            local = local + bias_shards[rank]
        local_activations.append(local)
    return {
        "local_activations": local_activations,
        "full_activation": torch.cat(local_activations, dim=-1),
    }
