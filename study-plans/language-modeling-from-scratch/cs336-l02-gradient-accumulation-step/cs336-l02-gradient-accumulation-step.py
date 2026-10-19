import torch

def gradient_accumulation_step(param, microbatch_inputs, microbatch_targets, lr):
    """
    Returns: dictionary containing new_param and full_grad tensors
    """
    work_param = param.detach().clone().requires_grad_(True)
    total_examples = sum(inputs.shape[0] for inputs in microbatch_inputs)

    for inputs, targets in zip(microbatch_inputs, microbatch_targets):
        predictions = inputs @ work_param
        loss = (predictions - targets).square().sum() / total_examples
        loss.backward()

    full_grad = work_param.grad.detach().clone()
    new_param = (work_param - lr * full_grad).detach()
    return {"new_param": new_param, "full_grad": full_grad}