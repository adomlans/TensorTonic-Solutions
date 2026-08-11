import torch


def _bn_relu(x, gamma, beta, mean, var, eps):
    C = x.shape[1]
    xn = (x - mean.view(1, C, 1, 1)) / torch.sqrt(var.view(1, C, 1, 1) + eps)
    out = xn * gamma.view(1, C, 1, 1) + beta.view(1, C, 1, 1)
    return torch.relu(out)


def _composite_layer(x, layer, eps):
    h = _bn_relu(x, layer["bn_gamma"], layer["bn_beta"],
                 layer["bn_mean"], layer["bn_var"], eps)
    return torch.nn.functional.conv2d(h, layer["conv_weight"], bias=None, padding=1)


def _dense_block(x, block, eps):
    feats = [x]
    cur = x
    for layer in block:
        feats.append(_composite_layer(cur, layer, eps))
        cur = torch.cat(feats, dim=1)
    return cur


def _transition(x, tr, eps):
    h = _bn_relu(x, tr["bn_gamma"], tr["bn_beta"],
                 tr["bn_mean"], tr["bn_var"], eps)
    out = torch.nn.functional.conv2d(h, tr["conv_weight"], bias=None)
    return torch.nn.functional.avg_pool2d(out, kernel_size=2, stride=2)


def densenet_forward(x, weights, growth_rate, eps=1e-5):
    """
    Return torch.Tensor of shape (N, num_classes): class logits from the full DenseNet.
    """
    feats = torch.nn.functional.conv2d(x, weights["stem_conv"], bias=None, padding=1)
    blocks = weights["blocks"]
    transitions = weights["transitions"]
    nb = len(blocks)
    for i, block in enumerate(blocks):
        feats = _dense_block(feats, block, eps)
        if i < nb - 1:
            feats = _transition(feats, transitions[i], eps)
    feats = _bn_relu(feats, weights["final_bn_gamma"], weights["final_bn_beta"],
                     weights["final_bn_mean"], weights["final_bn_var"], eps)
    pooled = feats.mean(dim=(2, 3))
    return pooled @ weights["fc_weight"].T + weights["fc_bias"]

