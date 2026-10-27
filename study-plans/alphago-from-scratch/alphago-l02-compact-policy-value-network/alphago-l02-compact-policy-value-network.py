import torch
import torch.nn.functional as F

def compact_policy_value_network(x, trunk_weight, trunk_scale, trunk_shift, residual_parameters, head_parameters, epsilon):
    """
    Returns: raw policy logits and tanh-bounded values.
    """
    def cast(value):
        return value.to(dtype=torch.float32, device=x.device)

    def batch_norm(value, scale, shift):
        mean = value.mean(dim=(0, 2, 3), keepdim=True)
        variance = (value - mean).square().mean(dim=(0, 2, 3), keepdim=True)
        normalized = (value - mean) * torch.rsqrt(variance + epsilon)
        return normalized * cast(scale).view(1, -1, 1, 1) + cast(shift).view(1, -1, 1, 1)

    features = F.conv2d(cast(x), cast(trunk_weight), padding=1)
    features = F.relu(batch_norm(features, trunk_scale, trunk_shift))

    for block in residual_parameters:
        residual = F.conv2d(features, cast(block["conv1_weight"]), padding=1)
        residual = F.relu(batch_norm(residual, block["scale1"], block["shift1"]))
        residual = F.conv2d(residual, cast(block["conv2_weight"]), padding=1)
        residual = batch_norm(residual, block["scale2"], block["shift2"])
        features = F.relu(features + residual)

    hp = head_parameters
    policy = F.conv2d(features, cast(hp["policy_conv_weight"]))
    policy = F.relu(batch_norm(policy, hp["policy_scale"], hp["policy_shift"])).flatten(1)
    policy_logits = F.linear(policy, cast(hp["policy_linear_weight"]), cast(hp["policy_linear_bias"]))

    value = F.conv2d(features, cast(hp["value_conv_weight"]))
    value = F.relu(batch_norm(value, hp["value_scale"], hp["value_shift"])).flatten(1)
    value = F.relu(F.linear(value, cast(hp["value_hidden_weight"]), cast(hp["value_hidden_bias"])))
    values = torch.tanh(F.linear(value, cast(hp["value_output_weight"]), cast(hp["value_output_bias"]))).squeeze(1)
    return policy_logits, values