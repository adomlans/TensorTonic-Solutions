import numpy as np

def patch_embed(image: np.ndarray, patch_size: int, embed_dim: int, W_proj: np.ndarray = None) -> np.ndarray:
    B, H, W, C = image.shape
    num_patches = (H // patch_size) * (W // patch_size)
    patch_dim = patch_size * patch_size * C

    patches = image.reshape(B, H // patch_size, patch_size, W // patch_size, patch_size, C)
    patches = patches.transpose(0, 1, 3, 2, 4, 5).reshape(B, num_patches, patch_dim)

    if W_proj is None:
        W_proj = np.random.randn(patch_dim, embed_dim) * 0.02
    else:
        W_proj = np.array(W_proj)
    return patches @ W_proj
