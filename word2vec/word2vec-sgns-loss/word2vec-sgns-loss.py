import torch
import torch.nn.functional as F

def sgns_loss(center_vec: torch.Tensor, pos_vec: torch.Tensor, neg_vecs: torch.Tensor) -> torch.Tensor:
    """
    Return a scalar torch.Tensor: the SGNS loss for one center/positive pair with k negatives.
    center_vec: (D,), pos_vec: (D,), neg_vecs: (k, D).
    """
    pos_score = torch.dot(center_vec, pos_vec)
    neg_scores = neg_vecs @ center_vec  # (k,)
    pos_loss = F.softplus(-pos_score)
    neg_loss = F.softplus(neg_scores).sum()
    return pos_loss + neg_loss
