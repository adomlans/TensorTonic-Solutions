import torch

def _read_depth_sources(sources, pseudo_query, eps):
    normalized = sources / torch.sqrt(sources.square().mean(dim=-1, keepdim=True) + eps)
    logits = (normalized * pseudo_query).sum(dim=-1)
    weights = torch.softmax(logits, dim=0)
    retrieved = (weights.unsqueeze(-1) * sources).sum(dim=0)
    return retrieved, weights

def block_attention_residual(embedding, previous_outputs, pseudo_query, block_size, eps=1e-6):
    """
    Returns: retrieved values, depth weights, and block-level sources.
    """
    layer_count = previous_outputs.shape[0]
    complete_layers = layer_count - layer_count % block_size
    completed = [
        previous_outputs[start:start + block_size].sum(dim=0)
        for start in range(0, complete_layers, block_size)
    ]
    sources = [embedding, *completed]
    if complete_layers < layer_count:
        sources.append(previous_outputs[complete_layers:].sum(dim=0))
    stacked_sources = torch.stack(sources)
    retrieved, weights = _read_depth_sources(stacked_sources, pseudo_query, eps)
    return retrieved, weights, stacked_sources
