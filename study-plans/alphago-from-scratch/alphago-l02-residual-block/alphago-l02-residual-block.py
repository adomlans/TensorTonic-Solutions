import torch
import torch.nn.functional as F

def alpha_go_residual_block(x, conv1_weight, scale1, shift1, conv2_weight, scale2, shift2, epsilon):
    """
    Returns: the torch.float32 residual-block output on the input device.
    """
    def cast(value):
        return value.to(dtype=torch.float32, device=x.device)

    def batch_norm(value, scale, shift):
        mean = value.mean(dim=(0, 2, 3), keepdim=True)
        variance = (value - mean).square().mean(dim=(0, 2, 3), keepdim=True)
        normalized = (value - mean) * torch.rsqrt(variance + epsilon)
        return normalized * cast(scale).view(1, -1, 1, 1) + cast(shift).view(1, -1, 1, 1)

    source = cast(x)
    first = F.conv2d(source, cast(conv1_weight), padding=1)
    first = F.relu(batch_norm(first, scale1, shift1))
    second = F.conv2d(first, cast(conv2_weight), padding=1)
    second = batch_norm(second, scale2, shift2)
    return F.relu(source + second)