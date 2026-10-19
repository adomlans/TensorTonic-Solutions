import math
import torch
import torch.nn.functional as F

def parameter_matched_swiglu(x, w_g, w_v, w_o, base_params):
    """
    Returns: dictionary containing output, hidden_width, and parameter_count
    """
    model_width = x.shape[-1]
    maximum_width = w_g.shape[1]
    hidden_width = math.floor(base_params / (3 * model_width) + 0.5)
    hidden_width = max(1, min(hidden_width, maximum_width))

    x_float = x.float()
    gate_weight = w_g[:, :hidden_width].float()
    value_weight = w_v[:, :hidden_width].float()
    output_weight = w_o[:hidden_width, :].float()
    gate = F.silu(x_float @ gate_weight)
    value = x_float @ value_weight
    output = ((gate * value) @ output_weight).to(x.dtype)

    return {
        "output": output,
        "hidden_width": hidden_width,
        "parameter_count": 3 * model_width * hidden_width,
    }