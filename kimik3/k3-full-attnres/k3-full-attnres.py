import torch

def full_attention_residual(embedding, previous_outputs, pseudo_query, eps=1e-6):
    """
    Returns: retrieved representations and depth-attention weights.
    """
    sources = torch.cat((embedding.unsqueeze(0), previous_outputs), dim=0)
    normalized_keys = sources / torch.sqrt(sources.square().mean(dim=-1, keepdim=True) + eps)
    logits = (normalized_keys * pseudo_query).sum(dim=-1)
    weights = torch.softmax(logits, dim=0)
    retrieved = (weights.unsqueeze(-1) * sources).sum(dim=0)
    return retrieved, weights
