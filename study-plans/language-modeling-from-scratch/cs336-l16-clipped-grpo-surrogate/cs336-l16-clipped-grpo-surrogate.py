import torch

def clipped_grpo_surrogate(new_log_probs, old_log_probs, advantages,
                           token_mask, clip_epsilon=0.2):
    """
    Returns: dictionary containing GRPO loss and clipping diagnostics
    """
    log_ratio = torch.clamp(
        new_log_probs - old_log_probs, min=-60.0, max=60.0
    )
    ratio = torch.exp(log_ratio)
    expanded_advantages = advantages[:, None]
    unclipped = ratio * expanded_advantages
    clipped_ratio = torch.clamp(
        ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon
    )
    surrogate = torch.minimum(
        unclipped, clipped_ratio * expanded_advantages
    )
    weights = token_mask.to(new_log_probs.dtype)
    counts = weights.sum(dim=1)
    sums = (surrogate * weights).sum(dim=1)
    per_sequence_surrogate = torch.where(
        counts > 0,
        sums / counts.clamp_min(1),
        torch.zeros_like(sums),
    )
    active_count = token_mask.sum()
    if int(active_count) == 0:
        zero = new_log_probs.new_zeros(())
        return {
            "loss": zero,
            "per_sequence_surrogate": per_sequence_surrogate,
            "clipped_fraction": zero,
        }
    clipped = (
        ((ratio < 1.0 - clip_epsilon) | (ratio > 1.0 + clip_epsilon))
        & token_mask
    )
    return {
        "loss": -(surrogate * weights).sum() / active_count,
        "per_sequence_surrogate": per_sequence_surrogate,
        "clipped_fraction": clipped.to(new_log_probs.dtype).sum() / active_count,
    }