import torch
import torch.nn.functional as F

def siglip_pairwise_loss(image_embeddings, text_embeddings,
                         logit_scale, bias):
    """
    Returns: dictionary containing SigLIP loss, logits, and pair labels
    """
    images = F.normalize(image_embeddings, dim=1)
    texts = F.normalize(text_embeddings, dim=1)
    logits = logit_scale * (images @ texts.transpose(0, 1)) + bias
    pair_labels = -torch.ones_like(logits)
    pair_labels.fill_diagonal_(1)
    return {
        "loss": -F.logsigmoid(pair_labels * logits).mean(),
        "logits": logits,
        "pair_labels": pair_labels,
    }