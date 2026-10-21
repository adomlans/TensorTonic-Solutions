import math
import torch

def kv_cache_decode_step(query, new_key, new_value, key_cache, value_cache,
                         num_query_heads, num_kv_heads):
    """
    Returns: dictionary containing output and updated caches
    """
    updated_keys = torch.cat((key_cache, new_key.unsqueeze(2)), dim=2)
    updated_values = torch.cat((value_cache, new_value.unsqueeze(2)), dim=2)
    group_size = num_query_heads // num_kv_heads
    grouped_query = query.float().reshape(
        query.shape[0], num_kv_heads, group_size, query.shape[-1]
    )
    scores = torch.einsum(
        "bhgd,bhsd->bhgs", grouped_query, updated_keys.float()
    ) / math.sqrt(query.shape[-1])
    weights = torch.softmax(scores, dim=-1)
    grouped_output = torch.einsum(
        "bhgs,bhsd->bhgd", weights, updated_values.float()
    )
    return {
        "output": grouped_output.reshape_as(query).to(query.dtype),
        "new_key_cache": updated_keys,
        "new_value_cache": updated_values,
    }
