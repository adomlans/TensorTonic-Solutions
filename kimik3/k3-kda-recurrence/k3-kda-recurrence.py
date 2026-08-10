import torch

def kda_recurrence(query, key, value, decay_logits, write_strength, output_gate_logits, output_projection, initial_state, g_min=-5.0, eps=1e-6):
    """
    Returns: sequence outputs and the final recurrent state.
    """
    state = initial_state.clone()
    outputs = []
    for position in range(query.shape[1]):
        query_t = query[:, position]
        key_t = key[:, position]
        value_t = value[:, position]
        beta_t = write_strength[:, position]
        alpha_t = torch.exp(g_min * torch.sigmoid(decay_logits[:, position]))
        decayed = alpha_t.unsqueeze(-1) * state
        recalled = (key_t.unsqueeze(-1) * decayed).sum(dim=-2)
        erase = beta_t.unsqueeze(-1) * key_t.unsqueeze(-1) * recalled.unsqueeze(-2)
        write = beta_t.unsqueeze(-1) * key_t.unsqueeze(-1) * value_t.unsqueeze(-2)
        state = decayed - erase + write
        read = (query_t.unsqueeze(-1) * state).sum(dim=-2)
        normalized = read / torch.sqrt(read.square().mean(dim=-1, keepdim=True) + eps)
        gated = torch.sigmoid(output_gate_logits[:, position]) * normalized
        merged = gated.reshape(gated.shape[0], -1)
        outputs.append(merged @ output_projection.transpose(0, 1))
    return torch.stack(outputs, dim=1), state
