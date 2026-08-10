import torch

def kda_context_parallel(transition_matrices, local_contributions, initial_state):
    """
    Returns: incoming states, outgoing states, and the final global state.
    """
    state_width = transition_matrices.shape[-1]
    prefix_transition = torch.eye(state_width, dtype=initial_state.dtype, device=initial_state.device)
    prefix_contribution = torch.zeros_like(initial_state)
    incoming_states = []
    outgoing_states = []
    for segment_index in range(transition_matrices.shape[0]):
        incoming = prefix_transition @ initial_state + prefix_contribution
        outgoing = transition_matrices[segment_index] @ incoming + local_contributions[segment_index]
        incoming_states.append(incoming)
        outgoing_states.append(outgoing)
        prefix_contribution = transition_matrices[segment_index] @ prefix_contribution + local_contributions[segment_index]
        prefix_transition = transition_matrices[segment_index] @ prefix_transition
    return torch.stack(incoming_states), torch.stack(outgoing_states), outgoing_states[-1]
