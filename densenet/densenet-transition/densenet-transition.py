import torch
import torch.nn.functional as F

def transition_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps=1e-5):
    x = torch.as_tensor(x, dtype=torch.float64)
    gamma = torch.as_tensor(bn_gamma, dtype=torch.float64)
    beta = torch.as_tensor(bn_beta, dtype=torch.float64)
    mean = torch.as_tensor(bn_mean, dtype=torch.float64)
    var = torch.as_tensor(bn_var, dtype=torch.float64)
    weight = torch.as_tensor(conv_weight, dtype=torch.float64)

    x_hat = (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + eps)
    y = F.relu(gamma[None, :, None, None] * x_hat + beta[None, :, None, None])
    y = F.conv2d(y, weight, bias=None, stride=1, padding=0)
    return F.avg_pool2d(y, kernel_size=2, stride=2)

