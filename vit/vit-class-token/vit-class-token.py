import numpy as np

def prepend_class_token(patches: np.ndarray, embed_dim: int, cls_token: np.ndarray = None) -> np.ndarray:
    B, N, D = patches.shape
    if cls_token is None:
        cls_token = np.random.randn(1, 1, D) * 0.02
    else:
        cls_token = np.array(cls_token)
    cls_tokens = np.tile(cls_token, (B, 1, 1))
    return np.concatenate([cls_tokens, patches], axis=1)
