import torch

def per_head_muon(parameter, gradient, previous_momentum, num_heads, momentum_coefficient, learning_rate):
    """
    Returns: updated parameter, momentum, and per-head orthogonalized update.
    """
    momentum = momentum_coefficient * previous_momentum + gradient
    rows_per_head = momentum.shape[0] // num_heads
    head_updates = []
    for head_index in range(num_heads):
        start = head_index * rows_per_head
        stop = start + rows_per_head
        block = momentum[start:stop]
        decomposition = block.float() if block.dtype in (torch.float16, torch.bfloat16) else block
        left, _, right_t = torch.linalg.svd(decomposition, full_matrices=False)
        head_updates.append((left @ right_t).to(block.dtype))
    orthogonalized = torch.cat(head_updates, dim=0)
    updated_parameter = parameter - learning_rate * orthogonalized
    return updated_parameter, momentum, orthogonalized
