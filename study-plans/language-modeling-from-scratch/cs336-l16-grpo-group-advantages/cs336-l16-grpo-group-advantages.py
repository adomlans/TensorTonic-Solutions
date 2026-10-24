import torch

def grpo_group_advantages(rewards, group_ids, epsilon=1e-8):
    """
    Returns: dictionary containing normalized advantages and active groups
    """
    if group_ids.numel() == 0:
        return {
            "advantages": rewards.clone(),
            "active_group_mask": torch.empty(
                0, dtype=torch.bool, device=rewards.device
            ),
            "active_group_count": 0,
        }
    group_count = int(group_ids.max()) + 1
    advantages = torch.zeros_like(rewards)
    active_group_mask = torch.zeros(
        group_count, dtype=torch.bool, device=rewards.device
    )
    for group_id in range(group_count):
        selected = group_ids == group_id
        values = rewards[selected]
        if torch.all(values == values[0]):
            continue
        centered = values - values.mean()
        standard_deviation = torch.sqrt(torch.mean(centered.square()))
        if float(standard_deviation) > 0.0:
            advantages[selected] = centered / (standard_deviation + epsilon)
            active_group_mask[group_id] = True
    return {
        "advantages": advantages,
        "active_group_mask": active_group_mask,
        "active_group_count": int(active_group_mask.sum()),
    }
