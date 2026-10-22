import torch

def multiple_choice_likelihood(logits, token_ids, continuation_mask):
    """
    Returns: dictionary containing continuation scores and selected choices
    """
    log_probabilities = torch.log_softmax(logits, dim=-1)
    target_scores = log_probabilities.gather(
        -1, token_ids.unsqueeze(-1)
    ).squeeze(-1)
    choice_scores = torch.where(
        continuation_mask, target_scores, torch.zeros_like(target_scores)
    ).sum(dim=-1)
    return {
        "choice_scores": choice_scores,
        "predicted_indices": torch.argmax(choice_scores, dim=-1),
    }