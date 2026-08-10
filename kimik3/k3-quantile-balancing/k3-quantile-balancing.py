import torch

def quantile_balancing(router_scores, current_bias, selected_count):
    """
    Returns: selected experts, mixture weights, loads, and the next centered bias.
    """
    biased_scores = router_scores + current_bias
    ranked = torch.argsort(biased_scores, dim=-1, descending=True, stable=True)
    selected = ranked[:, :selected_count]
    selected_raw = router_scores.gather(1, selected)
    mixture_weights = selected_raw / selected_raw.sum(dim=-1, keepdim=True)
    loads = torch.bincount(selected.reshape(-1), minlength=router_scores.shape[1])
    cutoffs = biased_scores.gather(1, ranked[:, selected_count:selected_count + 1])
    target_load = router_scores.shape[0] * selected_count // router_scores.shape[1]
    margins = router_scores - cutoffs
    ordered_margins = torch.sort(margins, dim=0, descending=True, stable=True).values
    uncentered_bias = -ordered_margins[target_load]
    next_bias = uncentered_bias - uncentered_bias.mean()
    return selected, mixture_weights, loads, next_bias
