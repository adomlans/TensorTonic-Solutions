import torch

def _situ_expert(inputs, gate_weight, up_weight, down_weight, gate_cap, up_cap):
    gate_values = inputs @ gate_weight.transpose(0, 1)
    up_values = inputs @ up_weight.transpose(0, 1)
    activation = (
        gate_cap * torch.tanh(gate_values / gate_cap) * torch.sigmoid(gate_values)
        * up_cap * torch.tanh(up_values / up_cap)
    )
    return activation @ down_weight.transpose(0, 1)

def stable_latent_moe(tokens, latent_down_projection, latent_up_projection, router_projection, current_bias, routed_gate_weights, routed_up_weights, routed_down_weights, shared_gate_weights, shared_up_weights, shared_down_weights, selected_count, eps=1e-6, gate_cap=4.0, up_cap=25.0):
    """
    Returns: final output, routes, mixture weights, and the latent routed aggregate.
    """
    latent = tokens @ latent_down_projection.transpose(0, 1)
    raw_scores = torch.sigmoid(tokens @ router_projection.transpose(0, 1))
    biased_scores = raw_scores + current_bias
    selected = torch.argsort(biased_scores, dim=-1, descending=True, stable=True)[:, :selected_count]
    selected_raw = raw_scores.gather(1, selected)
    mixture_weights = selected_raw / selected_raw.sum(dim=-1, keepdim=True)
    routed_aggregate = torch.zeros_like(latent)
    for token_index in range(tokens.shape[0]):
        for slot in range(selected_count):
            expert_index = int(selected[token_index, slot])
            expert_output = _situ_expert(
                latent[token_index:token_index + 1],
                routed_gate_weights[expert_index],
                routed_up_weights[expert_index],
                routed_down_weights[expert_index],
                gate_cap,
                up_cap,
            )[0]
            routed_aggregate[token_index] += mixture_weights[token_index, slot] * expert_output
    normalized = routed_aggregate / torch.sqrt(routed_aggregate.square().mean(dim=-1, keepdim=True) + eps)
    routed_output = normalized @ latent_up_projection.transpose(0, 1)
    shared_output = torch.zeros_like(tokens)
    for expert_index in range(shared_gate_weights.shape[0]):
        shared_output += _situ_expert(
            tokens,
            shared_gate_weights[expert_index],
            shared_up_weights[expert_index],
            shared_down_weights[expert_index],
            gate_cap,
            up_cap,
        )
    return shared_output + routed_output, selected, mixture_weights, routed_aggregate
