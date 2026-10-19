import torch

def rmsnorm(x, g, epsilon):
    """
    Returns: RMS-normalized tensor
    """
    x_float = x.float()
    scale_float = g.float()
    mean_square = x_float.square().mean(dim=-1, keepdim=True)
    normalized = x_float * torch.rsqrt(mean_square + epsilon)
    return (normalized * scale_float).to(x.dtype)