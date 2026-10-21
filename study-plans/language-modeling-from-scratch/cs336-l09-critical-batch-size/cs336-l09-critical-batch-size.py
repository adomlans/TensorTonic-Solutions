import torch

def critical_batch_size(batch_sizes, steps_to_target, reached_mask):
    """
    Returns: dictionary containing examples and the critical batch estimate
    """
    reached_batches = batch_sizes[reached_mask].double()
    reached_steps = steps_to_target[reached_mask].double()
    reached_examples = reached_batches * reached_steps
    examples = [
        float(batch_sizes[index].double() * steps_to_target[index].double()) if bool(reached_mask[index]) else None
        for index in range(batch_sizes.numel())
    ]
    min_steps = float(torch.min(reached_steps))
    min_examples = float(torch.min(reached_examples))
    return {
        "examples": examples,
        "min_steps": min_steps,
        "min_examples": min_examples,
        "critical_batch_size": float(min_examples / min_steps),
    }