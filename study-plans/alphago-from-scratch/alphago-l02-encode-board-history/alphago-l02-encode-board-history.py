import torch

def encode_go_history(history, to_play):
    """
    Returns: a floating 17-plane current-player-relative history tensor.
    """
    device = history[0].device
    size = history[0].shape[0]
    planes = []

    for index in range(8):
        if index < len(history):
            board = history[index]
            planes.append((board == to_play).to(dtype=torch.float32, device=device))
            planes.append((board == -to_play).to(dtype=torch.float32, device=device))
        else:
            planes.append(torch.zeros((size, size), dtype=torch.float32, device=device))
            planes.append(torch.zeros((size, size), dtype=torch.float32, device=device))

    colour = torch.full(
        (size, size),
        1.0 if to_play == 1 else 0.0,
        dtype=torch.float32,
        device=device,
    )
    planes.append(colour)
    return torch.stack(planes, dim=0)