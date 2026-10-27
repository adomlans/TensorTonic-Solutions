import torch
import torch.nn.functional as F

def policy_value_heads(features, policy_conv_weight, policy_scale, policy_shift, policy_linear_weight, policy_linear_bias, value_conv_weight, value_scale, value_shift, value_hidden_weight, value_hidden_bias, value_output_weight, value_output_bias, epsilon):
    """
    Returns: raw policy logits and one tanh-bounded value per batch item.
    """
    def cast(value):
        return value.to(dtype=torch.float32, device=features.device)

    def batch_norm(value, scale, shift):
        mean = value.mean(dim=(0, 2, 3), keepdim=True)
        variance = (value - mean).square().mean(dim=(0, 2, 3), keepdim=True)
        normalized = (value - mean) * torch.rsqrt(variance + epsilon)
        return normalized * cast(scale).view(1, -1, 1, 1) + cast(shift).view(1, -1, 1, 1)

    source = cast(features)
    policy = F.conv2d(source, cast(policy_conv_weight))
    policy = F.relu(batch_norm(policy, policy_scale, policy_shift)).flatten(1)
    logits = F.linear(policy, cast(policy_linear_weight), cast(policy_linear_bias))

    value = F.conv2d(source, cast(value_conv_weight))
    value = F.relu(batch_norm(value, value_scale, value_shift)).flatten(1)
    value = F.relu(F.linear(value, cast(value_hidden_weight), cast(value_hidden_bias)))
    value = F.linear(value, cast(value_output_weight), cast(value_output_bias)).squeeze(-1)
    return logits, torch.tanh(value)