import numpy as np

def add_position_embedding(patches: np.ndarray, num_patches: int, embed_dim: int, pos_embed: np.ndarray = None) -> np.ndarray:
    B, N, D = patches.shape
    if pos_embed is None:
        pos_embed = np.random.randn(1, N, D) * 0.02
    else:
        pos_embed = np.array(pos_embed)
    return patches + pos_embed
