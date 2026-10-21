import math
import torch

def wsd_learning_rate(total_steps, warmup_steps, stable_steps,
                      peak_learning_rate, final_ratio):
    """
    Returns: float64 learning-rate tensor
    """
    rates = [
        peak_learning_rate * (step + 1) / warmup_steps
        for step in range(warmup_steps)
    ]
    rates.extend([peak_learning_rate] * stable_steps)
    decay_steps = total_steps - warmup_steps - stable_steps
    for index in range(decay_steps):
        progress = 1.0 if decay_steps == 1 else index / (decay_steps - 1)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        multiplier = final_ratio + (1 - final_ratio) * cosine
        rates.append(peak_learning_rate * multiplier)
    return torch.tensor(rates, dtype=torch.float64)
