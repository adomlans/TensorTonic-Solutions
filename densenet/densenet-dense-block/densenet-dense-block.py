import torch
import torch.nn.functional as F

def dense_block(x, layers, growth_rate, eps=1e-5):
    x = torch.as_tensor(x, dtype=torch.float64)
    feats = [x]
    for p in layers:
        # input to this layer is the channel-wise concat of x and all previous outputs
        inp = torch.cat(feats, dim=1)
        gamma = torch.as_tensor(p['bn_gamma'], dtype=torch.float64)
        beta = torch.as_tensor(p['bn_beta'], dtype=torch.float64)
        mean = torch.as_tensor(p['bn_mean'], dtype=torch.float64)
        var = torch.as_tensor(p['bn_var'], dtype=torch.float64)
        weight = torch.as_tensor(p['conv_weight'], dtype=torch.float64)
        # BN (inference) -> ReLU -> 3x3 Conv (stride 1, padding 1) producing growth_rate channels
        xhat = (inp - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + eps)
        z = F.relu(gamma[None, :, None, None] * xhat + beta[None, :, None, None])
        out = F.conv2d(z, weight, bias=None, stride=1, padding=1)
        feats.append(out)
    # block output: concat of input and ALL L layer outputs
    return torch.cat(feats, dim=1)
pass
