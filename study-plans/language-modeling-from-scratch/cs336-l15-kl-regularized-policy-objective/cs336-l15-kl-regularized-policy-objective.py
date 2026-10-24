import torch

def kl_regularized_policy_objective(policy_log_probs, reference_log_probs,
                                    rewards, response_mask, beta):
    """
    Returns: dictionary containing objective, penalized rewards, and divergence
    """
    differences = (policy_log_probs - reference_log_probs).float()
    sampled_divergences = torch.where(
        response_mask, differences, torch.zeros_like(differences)
    ).sum(dim=1)
    penalized_rewards = rewards.float() - beta * sampled_divergences
    objective = (
        penalized_rewards.mean()
        if penalized_rewards.numel() > 0
        else torch.zeros((), dtype=torch.float32, device=rewards.device)
    )
    return {
        "objective": objective,
        "penalized_rewards": penalized_rewards,
        "sampled_divergences": sampled_divergences,
    }