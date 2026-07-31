def linear_lr(step, total_steps, initial_lr, final_lr=0.0, warmup_steps=0) -> float:
    """
    Linear warmup (0→initial_lr) then linear decay (initial_lr→final_lr).
    Steps are 0-based; clamp at final_lr after total_steps.
    """
    # Write code here
    if total_steps <= 0:
        return float(final_lr)
    step = max(0, int(step))
    if warmup_steps > 0 and step < warmup_steps:
        return float(initial_lr * (step / warmup_steps))
    s_eff = min(step, total_steps)
    decay_len = max(total_steps - warmup_steps, 1)
    return float(final_lr + (initial_lr - final_lr) * max(0.0, (total_steps - s_eff) / decay_len))
    pass