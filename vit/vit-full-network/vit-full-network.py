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

class VisionTransformer:
    def __init__(self, image_size=224, patch_size=16, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 W_patch=None, cls_token=None, pos_embed=None,
                 encoder_weights=None, W_head=None):
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = patch_size * patch_size * 3

        if W_patch is not None:
            self.W_patch = np.array(W_patch)
            self.cls_token = np.array(cls_token)
            self.pos_embed = np.array(pos_embed)
            self.encoder_weights = [{k: np.array(v) for k, v in bw.items()} for bw in encoder_weights]
            self.W_head = np.array(W_head)
        else:
            self.W_patch = np.random.randn(patch_dim, embed_dim) * 0.02
            self.cls_token = np.random.randn(1, 1, embed_dim) * 0.02
            self.pos_embed = np.random.randn(1, self.num_patches + 1, embed_dim) * 0.02
            self.encoder_weights = []
            hd = int(embed_dim * mlp_ratio)
            for _ in range(depth):
                bw = {
                    'Wq': np.random.randn(embed_dim, embed_dim) * 0.02,
                    'Wk': np.random.randn(embed_dim, embed_dim) * 0.02,
                    'Wv': np.random.randn(embed_dim, embed_dim) * 0.02,
                    'Wo': np.random.randn(embed_dim, embed_dim) * 0.02,
                    'W1': np.random.randn(embed_dim, hd) * 0.02,
                    'W2': np.random.randn(hd, embed_dim) * 0.02,
                }
                self.encoder_weights.append(bw)
            self.W_head = np.random.randn(embed_dim, num_classes) * 0.02

    def forward(self, x):
        B, H, W, C = x.shape
        N = self.num_patches
        D = self.embed_dim

        patch_dim = self.patch_size * self.patch_size * C
        patches = x.reshape(B, H // self.patch_size, self.patch_size,
                           W // self.patch_size, self.patch_size, C)
        patches = patches.transpose(0, 1, 3, 2, 4, 5).reshape(B, N, patch_dim)
        z = patches @ self.W_patch

        cls_tokens = np.tile(self.cls_token, (B, 1, 1))
        z = np.concatenate([cls_tokens, z], axis=1)
        z = z + self.pos_embed

        for bw in self.encoder_weights:
            norm1 = layer_norm(z)
            Q = norm1 @ bw['Wq']; K = norm1 @ bw['Wk']; V = norm1 @ bw['Wv']
            hd = D // self.num_heads
            Q = Q.reshape(B, -1, self.num_heads, hd).transpose(0, 2, 1, 3)
            K = K.reshape(B, -1, self.num_heads, hd).transpose(0, 2, 1, 3)
            V = V.reshape(B, -1, self.num_heads, hd).transpose(0, 2, 1, 3)
            attn = softmax(Q @ K.transpose(0, 1, 3, 2) / np.sqrt(hd))
            ao = (attn @ V).transpose(0, 2, 1, 3).reshape(B, -1, D)
            ao = ao @ bw['Wo']
            z = z + ao
            norm2 = layer_norm(z)
            mo = gelu(norm2 @ bw['W1']) @ bw['W2']
            z = z + mo

        cls_out = z[:, 0]
        mean = cls_out.mean(axis=-1, keepdims=True)
        std = cls_out.std(axis=-1, keepdims=True)
        norm = (cls_out - mean) / (std + 1e-6)
        return norm @ self.W_head
