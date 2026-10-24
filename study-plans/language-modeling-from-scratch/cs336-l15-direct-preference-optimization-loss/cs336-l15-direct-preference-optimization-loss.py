import torch
import torch.nn.functional as F

def direct_preference_optimization_loss(policy_chosen, policy_rejected,
                                        reference_chosen, reference_rejected,
                                        beta, label_smoothing=0.0):
    """
    Returns: dictionary containing DPO losses, rewards, and accuracy
    """
    logits = beta * (
        (policy_chosen - policy_rejected)
        - (reference_chosen - reference_rejected)
    )
    per_example_loss = -(
        (1.0 - label_smoothing) * F.logsigmoid(logits)
        + label_smoothing * F.logsigmoid(-logits)
    )
    chosen_rewards = beta * (policy_chosen - reference_chosen)
    rejected_rewards = beta * (policy_rejected - reference_rejected)
    preference_accuracy = (chosen_rewards > rejected_rewards).to(
        policy_chosen.dtype
    ).mean()
    return {
        "loss": per_example_loss.mean(),
        "per_example_loss": per_example_loss,
        "chosen_rewards": chosen_rewards,
        "rejected_rewards": rejected_rewards,
        "preference_accuracy": preference_accuracy,
    }