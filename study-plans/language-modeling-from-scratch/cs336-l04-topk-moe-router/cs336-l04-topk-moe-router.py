import torch

def topk_moe_router(x, w_router, expert_outputs, shared_output, k):
    """
    Returns: dictionary containing routed output, selections, weights, and load statistics
    """
    scores = x.float() @ w_router.float()
    selected_experts = torch.argsort(
        scores, dim=-1, descending=True, stable=True
    )[:, :k]
    selected_scores = torch.gather(scores, 1, selected_experts)
    selected_weights = torch.softmax(selected_scores, dim=-1)

    output_width = expert_outputs.shape[-1]
    expanded_indices = selected_experts.unsqueeze(-1).expand(-1, -1, output_width)
    selected_outputs = torch.gather(expert_outputs.float(), 1, expanded_indices)
    routed = (selected_outputs * selected_weights.unsqueeze(-1)).sum(dim=1)

    selection_mask = torch.zeros(
        x.shape[0], w_router.shape[1], dtype=torch.bool, device=x.device
    )
    selection_mask.scatter_(1, selected_experts, True)
    token_fraction = selection_mask.float().mean(dim=0)
    probability_mass = torch.softmax(scores, dim=-1).mean(dim=0)

    return {
        "output": (shared_output.float() + routed).to(x.dtype),
        "selected_experts": selected_experts,
        "selected_weights": selected_weights.to(x.dtype),
        "token_fraction": token_fraction.to(x.dtype),
        "probability_mass": probability_mass.to(x.dtype),
    }
