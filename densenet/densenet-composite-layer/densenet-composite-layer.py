import torch
import torch.nn.functional as F

def composite_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps=1e-5):
    """
    Return torch.Tensor of shape (N, growth_rate, H, W): BN, ReLU, then a 3x3 same-padding convolution.
    """
    x = torch.as_tensor(x, dtype=torch.float64)
    g = torch.as_tensor(bn_gamma, dtype=torch.float64)
    b = torch.as_tensor(bn_beta, dtype=torch.float64)
    m = torch.as_tensor(bn_mean, dtype=torch.float64)
    v = torch.as_tensor(bn_var, dtype=torch.float64)
    w = torch.as_tensor(conv_weight, dtype=torch.float64)
    xh = (x - m[None, :, None, None]) / torch.sqrt(v[None, :, None, None] + eps)
    y = g[None, :, None, None] * xh + b[None, :, None, None]
    y = F.relu(y)
    return F.conv2d(y, w, bias=None, stride=1, padding=1)

