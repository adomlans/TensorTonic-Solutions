import torch
import torch.nn.functional as F

def policy_value_training_step(features, policy_weight, policy_bias, value_weight, value_bias, target_policies, target_values, learning_rate, l2_coefficient):
    x = features.to(dtype=torch.float32)
    policy_targets = target_policies.to(dtype=torch.float32)
    value_targets = target_values.to(dtype=torch.float32)
    pw = policy_weight.to(dtype=torch.float32).detach().clone().requires_grad_(True)
    pb = policy_bias.to(dtype=torch.float32).detach().clone().requires_grad_(True)
    vw = value_weight.to(dtype=torch.float32).detach().clone().requires_grad_(True)
    original_value_bias_shape = value_bias.shape
    vb = value_bias.to(dtype=torch.float32).reshape(1).detach().clone().requires_grad_(True)

    logits = F.linear(x, pw, pb)
    values = torch.tanh(F.linear(x, vw.unsqueeze(0), vb).squeeze(1))
    policy_loss = -(policy_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    value_loss = F.mse_loss(values, value_targets)
    regularization = l2_coefficient * sum(parameter.square().sum() for parameter in (pw, pb, vw, vb))
    loss = policy_loss + value_loss + regularization

    gradients = torch.autograd.grad(loss, (pw, pb, vw, vb))
    updated = tuple((parameter - learning_rate * gradient).detach() for parameter, gradient in zip((pw, pb, vw, vb), gradients))
    return loss.detach(), updated[0], updated[1], updated[2], updated[3].reshape(original_value_bias_shape)