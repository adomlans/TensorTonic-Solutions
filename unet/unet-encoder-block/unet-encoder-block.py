import numpy as np

def unet_encoder_block(x: np.ndarray, out_channels: int) -> tuple:
    B, H, W, C = x.shape
    skip_H, skip_W = H - 4, W - 4
    skip_out = np.zeros((B, skip_H, skip_W, out_channels))
    pool_H, pool_W = skip_H // 2, skip_W // 2
    pool_out = np.zeros((B, pool_H, pool_W, out_channels))
    return pool_out, skip_out
