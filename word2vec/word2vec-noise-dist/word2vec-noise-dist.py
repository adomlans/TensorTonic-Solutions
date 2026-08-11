import torch

def noise_distribution(counts: torch.Tensor, alpha: float = 0.75) -> torch.Tensor:
    """
    Return torch.Tensor of shape (vocab_size,), a probability distribution that sums to 1.
    """
    counts = torch.as_tensor(counts, dtype=torch.float64)
    w = counts ** alpha
    return w / w.sum()
