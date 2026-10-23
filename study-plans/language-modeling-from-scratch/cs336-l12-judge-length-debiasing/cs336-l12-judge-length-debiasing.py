import torch

def judge_length_debiasing(num_models, model_a, model_b,
                            length_differences, outcomes,
                            steps, learning_rate):
    """
    Returns: dictionary containing fitted bias and debiased probabilities
    """
    utilities = torch.zeros(num_models, dtype=torch.float64, device=outcomes.device)
    length_coefficient = torch.zeros((), dtype=torch.float64, device=outcomes.device)
    length_values = length_differences.double()
    outcome_values = outcomes.double()
    for _ in range(steps):
        logits = (
            utilities[model_a] - utilities[model_b]
            + length_coefficient * length_values
        )
        residuals = outcome_values - torch.sigmoid(logits)
        utility_gradient = torch.zeros_like(utilities)
        if residuals.numel() > 0:
            utility_gradient.index_add_(0, model_a, residuals)
            utility_gradient.index_add_(0, model_b, -residuals)
            utility_gradient /= residuals.numel()
            length_coefficient = length_coefficient + learning_rate * torch.mean(
                residuals * length_values
            )
        utilities = utilities + learning_rate * utility_gradient
        utilities = utilities - utilities.mean()

    debiased_probabilities = torch.sigmoid(
        utilities[model_a] - utilities[model_b]
    )
    wins = torch.zeros_like(utilities)
    counts = torch.zeros_like(utilities)
    if debiased_probabilities.numel() > 0:
        wins.index_add_(0, model_a, debiased_probabilities)
        wins.index_add_(0, model_b, 1.0 - debiased_probabilities)
        counts.index_add_(0, model_a, torch.ones_like(debiased_probabilities))
        counts.index_add_(0, model_b, torch.ones_like(debiased_probabilities))
    model_win_rates = torch.where(
        counts > 0,
        wins / counts.clamp_min(1),
        torch.full_like(wins, 0.5),
    )
    return {
        "length_coefficient": float(length_coefficient),
        "debiased_probabilities": debiased_probabilities,
        "model_win_rates": model_win_rates,
    }
