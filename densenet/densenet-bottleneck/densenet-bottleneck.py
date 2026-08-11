import torch
import torch.nn.functional as F

def bottleneck_layer(x, bn1_gamma, bn1_beta, bn1_mean, bn1_var, conv1_weight,
                     bn2_gamma, bn2_beta, bn2_mean, bn2_var, conv2_weight, eps=1e-5):
    x = torch.as_tensor(x, dtype=torch.float64)

    def bn(t, g, b, m, v):
        g = torch.as_tensor(g, dtype=torch.float64)
        b = torch.as_tensor(b, dtype=torch.float64)
        m = torch.as_tensor(m, dtype=torch.float64)
        v = torch.as_tensor(v, dtype=torch.float64)
        xh = (t - m[None, :, None, None]) / torch.sqrt(v[None, :, None, None] + eps)
        return g[None, :, None, None] * xh + b[None, :, None, None]

    # Stage 1: BN-ReLU-Conv(1x1) -> 4k channels
    y = F.relu(bn(x, bn1_gamma, bn1_beta, bn1_mean, bn1_var))
    y = F.conv2d(y, torch.as_tensor(conv1_weight, dtype=torch.float64), bias=None, stride=1, padding=0)

    # Stage 2: BN-ReLU-Conv(3x3) -> k channels
    y = F.relu(bn(y, bn2_gamma, bn2_beta, bn2_mean, bn2_var))
    y = F.conv2d(y, torch.as_tensor(conv2_weight, dtype=torch.float64), bias=None, stride=1, padding=1)
    return y
