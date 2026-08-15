import numpy as np

def gelu(x):
    return x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0/np.pi) * (x + 0.044715 * np.power(x, 3))))

def layer_norm(x, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)

def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def vit_encoder_block(x, embed_dim, num_heads, mlp_ratio=4.0,
                      Wq=None, Wk=None, Wv=None, Wo=None, W1=None, W2=None):
    B, N, D = x.shape
    if Wq is None:
        Wq = np.random.randn(D, D) * 0.02
        Wk = np.random.randn(D, D) * 0.02
        Wv = np.random.randn(D, D) * 0.02
        Wo = np.random.randn(D, D) * 0.02
        hd_dim = int(D * mlp_ratio)
        W1 = np.random.randn(D, hd_dim) * 0.02
        W2 = np.random.randn(hd_dim, D) * 0.02
    else:
        Wq = np.array(Wq); Wk = np.array(Wk); Wv = np.array(Wv)
        Wo = np.array(Wo); W1 = np.array(W1); W2 = np.array(W2)

    norm1 = layer_norm(x)
    Q = norm1 @ Wq; K = norm1 @ Wk; V = norm1 @ Wv
    head_dim = D // num_heads
    Q = Q.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
    attn = softmax(Q @ K.transpose(0, 1, 3, 2) / np.sqrt(head_dim))
    attn_out = (attn @ V).transpose(0, 2, 1, 3).reshape(B, N, D)
    attn_out = attn_out @ Wo
    x = x + attn_out

    norm2 = layer_norm(x)
    mlp_out = gelu(norm2 @ W1) @ W2
    x = x + mlp_out
    return x
