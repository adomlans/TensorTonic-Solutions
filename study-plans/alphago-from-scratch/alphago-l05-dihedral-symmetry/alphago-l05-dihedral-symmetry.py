import numpy as np

def apply_go_symmetry(state_planes, policy, value, transform_id):
    state = np.asarray(state_planes)
    probabilities = np.asarray(policy)
    board_size = state.shape[-1]

    def transform(array):
        if transform_id < 4:
            return np.rot90(array, k=transform_id, axes=(-2, -1))
        reflected = np.flip(array, axis=-1)
        return np.rot90(reflected, k=transform_id - 4, axes=(-2, -1))

    transformed_state = np.array(transform(state), copy=True)
    board_policy = probabilities[:-1].reshape(board_size, board_size)
    transformed_board = np.array(transform(board_policy), copy=True).reshape(-1)
    transformed_policy = np.concatenate((transformed_board, probabilities[-1:].copy()))
    return transformed_state, transformed_policy, value
