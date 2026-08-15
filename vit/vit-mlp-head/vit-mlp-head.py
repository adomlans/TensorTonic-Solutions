import numpy as np

def classification_head(encoder_output: np.ndarray, num_classes: int, W_head: np.ndarray = None) -> np.ndarray:
    cls_token = encoder_output[:, 0]
    mean = cls_token.mean(axis=-1, keepdims=True)
    std = cls_token.std(axis=-1, keepdims=True)
    norm = (cls_token - mean) / (std + 1e-6)
    if W_head is None:
        W_head = np.random.randn(norm.shape[1], num_classes) * 0.02
    else:
        W_head = np.array(W_head)
    return norm @ W_head
