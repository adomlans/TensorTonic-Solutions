import math
import torch

def attention_scores(q, k, num_heads):
    """
    Returns: tensor of shape (batch, heads, query_length, key_length)
    """
    batch_size, query_length, model_width = q.shape
    key_length = k.shape[1]
    head_width = model_width // num_heads

    q_heads = q.reshape(batch_size, query_length, num_heads, head_width).transpose(1, 2)
    k_heads = k.reshape(batch_size, key_length, num_heads, head_width).transpose(1, 2)
    scores = torch.einsum("bhqd,bhkd->bhqk", q_heads, k_heads)
    return scores / math.sqrt(head_width)