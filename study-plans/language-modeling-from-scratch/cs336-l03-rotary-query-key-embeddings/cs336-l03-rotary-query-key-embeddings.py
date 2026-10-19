import torch

def rotary_embed(q, k, positions, inv_freq):
    """
    Returns: dictionary containing q_rotated and k_rotated
    """
    if positions.ndim == 1:
        position_values = positions.float().reshape(1, 1, -1, 1)
    else:
        position_values = positions.float().reshape(positions.shape[0], 1, positions.shape[1], 1)

    angles = position_values * inv_freq.float().reshape(1, 1, 1, -1)
    cosine = torch.cos(angles)
    sine = torch.sin(angles)

    def rotate(tensor):
        tensor_float = tensor.float()
        even = tensor_float[..., 0::2]
        odd = tensor_float[..., 1::2]
        rotated_even = even * cosine - odd * sine
        rotated_odd = even * sine + odd * cosine
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2).to(tensor.dtype)

    return {"q_rotated": rotate(q), "k_rotated": rotate(k)}
