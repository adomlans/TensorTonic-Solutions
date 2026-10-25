import torch
import torch.nn.functional as F

def symmetric_clip_loss(image_embeddings, text_embeddings, temperature):
    """
    Returns: dictionary containing symmetric loss, similarities, and accuracies
    """
    images = F.normalize(image_embeddings, dim=1)
    texts = F.normalize(text_embeddings, dim=1)
    similarities = images @ texts.transpose(0, 1) / temperature
    targets = torch.arange(image_embeddings.shape[0], device=image_embeddings.device)
    image_loss = F.cross_entropy(similarities, targets)
    text_loss = F.cross_entropy(similarities.transpose(0, 1), targets)
    return {
        "loss": (image_loss + text_loss) / 2,
        "similarities": similarities,
        "image_to_text_accuracy": (similarities.argmax(dim=1) == targets).to(
            image_embeddings.dtype
        ).mean(),
        "text_to_image_accuracy": (similarities.argmax(dim=0) == targets).to(
            image_embeddings.dtype
        ).mean(),
    }