import torch

def simulate_collectives(rank_tensors, collective):
    """
    Returns: list containing one output tensor per rank
    """
    world_size = len(rank_tensors)
    if collective == "all_gather":
        gathered = torch.cat(rank_tensors, dim=0)
        return [gathered.clone() for _ in range(world_size)]

    reduced = torch.stack(rank_tensors, dim=0).sum(dim=0)
    if collective == "all_reduce":
        return [reduced.clone() for _ in range(world_size)]
    if collective == "reduce_scatter":
        return [chunk.clone() for chunk in torch.chunk(reduced, world_size)]

    source_chunks = [torch.chunk(tensor, world_size) for tensor in rank_tensors]
    return [
        torch.cat([source_chunks[source][destination] for source in range(world_size)])
        for destination in range(world_size)
    ]
