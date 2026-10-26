import torch

def squared_error_loss_gradients(predictions, targets):
    """
    Returns: the summed squared-error loss and prediction gradients
    """
    errors = predictions - targets
    loss = (errors * errors).sum()
    gradients = 2 * errors
    return (loss, gradients)