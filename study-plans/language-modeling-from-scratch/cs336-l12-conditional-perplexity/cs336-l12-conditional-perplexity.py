import math
import torch

def conditional_perplexity(logits, target_ids, response_mask):
    """
    Returns: dictionary containing response-token NLL, perplexity, and token count
    """
    selected_token_count = int(response_mask.sum())
    if selected_token_count == 0:
        return {
            "negative_log_likelihood": 0.0,
            "perplexity": 1.0,
            "selected_token_count": 0,
        }
    log_probabilities = torch.log_softmax(logits, dim=-1)
    target_log_probabilities = log_probabilities.gather(
        -1, target_ids.unsqueeze(-1)
    ).squeeze(-1)
    negative_log_likelihood = float(
        -target_log_probabilities[response_mask].double().mean()
    )
    return {
        "negative_log_likelihood": negative_log_likelihood,
        "perplexity": math.exp(negative_log_likelihood),
        "selected_token_count": selected_token_count,
    }
