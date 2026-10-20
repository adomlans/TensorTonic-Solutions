import torch

def data_parallel_gradient_sync(parameter_replicas, gradient_replicas, learning_rate):
    """
    Returns: dictionary containing mean gradients and updated replicas
    """
    world_size = len(parameter_replicas)
    parameter_count = len(parameter_replicas[0])
    mean_gradients = []
    for parameter_index in range(parameter_count):
        reference = parameter_replicas[0][parameter_index]
        total = torch.zeros_like(reference, dtype=torch.float32)
        for rank in range(world_size):
            gradient = gradient_replicas[rank][parameter_index]
            if gradient is not None:
                total = total + gradient.float()
        mean_gradients.append((total / world_size).to(reference.dtype))

    updated_replicas = []
    for _ in range(world_size):
        updated_replicas.append([
            (parameter_replicas[0][index].float() - learning_rate * mean_gradients[index].float()).to(parameter_replicas[0][index].dtype).clone()
            for index in range(parameter_count)
        ])
    return {"mean_gradients": mean_gradients, "updated_replicas": updated_replicas}
