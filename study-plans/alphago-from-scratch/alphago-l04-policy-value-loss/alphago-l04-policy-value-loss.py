import torch

def alpha_go_policy_value_loss(policy_logits, predicted_values, target_policies, target_values, parameters, l2_coefficient):
    """
    Returns: Total, policy, value, and L2 regularization scalar tensors.
    """
    log_probabilities = torch.log_softmax(policy_logits, dim=1)
    policy_loss = -(target_policies * log_probabilities).sum(dim=1).mean()

    value_predictions = predicted_values.reshape(-1)
    value_targets = target_values.reshape(-1)
    value_loss = (value_predictions - value_targets).square().mean()

    regularization_loss = policy_logits.new_zeros(())
    for parameter in parameters:
        regularization_loss = regularization_loss + parameter.square().sum()
    regularization_loss = regularization_loss * l2_coefficient

    total_loss = policy_loss + value_loss + regularization_loss
    return total_loss, policy_loss, value_loss, regularization_loss