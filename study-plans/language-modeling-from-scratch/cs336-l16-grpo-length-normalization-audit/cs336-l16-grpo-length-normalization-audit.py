import torch

def grpo_length_normalization_audit(token_log_probs, advantages,
                                    token_mask):
    """
    Returns: dictionary containing sequence and token normalization results
    """
    weights = token_mask.to(token_log_probs.dtype)
    token_sums = (token_log_probs * weights).sum(dim=1)
    counts = weights.sum(dim=1)
    sequence_summed = advantages * token_sums
    token_averaged = torch.where(
        counts > 0,
        sequence_summed / counts.clamp_min(1),
        torch.zeros_like(sequence_summed),
    )
    denominator = token_averaged.abs().sum()
    magnitude_ratio = torch.where(
        denominator > 0,
        sequence_summed.abs().sum() / denominator,
        torch.zeros_like(denominator),
    )
    return {
        "sequence_summed": sequence_summed,
        "token_averaged": token_averaged,
        "magnitude_ratio": magnitude_ratio,
    }