import torch

def speculative_sampling(draft_tokens, draft_probabilities, target_probabilities,
                         acceptance_random_values, residual_random_value):
    """
    Returns: dictionary containing emitted tokens and acceptance information
    """
    emitted = []
    for index in range(draft_tokens.numel()):
        token = int(draft_tokens[index])
        draft_probability = float(draft_probabilities[index, token])
        target_probability = float(target_probabilities[index, token])
        if draft_probability == 0:
            threshold = 1.0 if target_probability > 0 else 0.0
        else:
            threshold = min(1.0, target_probability / draft_probability)

        if float(acceptance_random_values[index]) < threshold:
            emitted.append(token)
            continue

        residual = torch.clamp(
            target_probabilities[index].double() - draft_probabilities[index].double(),
            min=0,
        )
        cumulative = torch.cumsum(residual / residual.sum(), dim=0)
        random_value = torch.tensor(
            residual_random_value, dtype=torch.float64, device=cumulative.device
        )
        replacement = int(torch.searchsorted(cumulative, random_value, right=True))
        last_positive = int(torch.nonzero(residual > 0)[-1, 0])
        emitted.append(min(replacement, last_positive))
        return {
            "tokens": torch.tensor(emitted, dtype=torch.int64, device=draft_tokens.device),
            "accepted_count": index,
            "rejected_at": index,
        }

    return {
        "tokens": draft_tokens.clone(),
        "accepted_count": draft_tokens.numel(),
        "rejected_at": None,
    }