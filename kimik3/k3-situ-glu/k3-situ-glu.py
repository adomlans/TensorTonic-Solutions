import torch

def situ_glu(input_tensor, gate_projection, up_projection, gate_cap=4.0, up_cap=25.0):
    """
    Returns: the bounded element-wise gated activation.
    """
    gate_values = input_tensor @ gate_projection.transpose(0, 1)
    up_values = input_tensor @ up_projection.transpose(0, 1)
    gate_branch = gate_cap * torch.tanh(gate_values / gate_cap) * torch.sigmoid(gate_values)
    up_branch = up_cap * torch.tanh(up_values / up_cap)
    return gate_branch * up_branch
