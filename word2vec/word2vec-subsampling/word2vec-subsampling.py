import torch

def subsample_keep_probs(counts: torch.Tensor, t: float = 1e-5) -> torch.Tensor:
    counts = torch.as_tensor(counts, dtype=torch.float64)
    total = counts.sum()
    f = counts / total
    keep = torch.sqrt(t / f)
    return torch.clamp(keep, max=1.0)

