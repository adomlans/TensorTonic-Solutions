import torch

def hyperparameter_scaling_fit(model_scales, learning_rates, batch_sizes,
                               measured_losses, target_scale):
    """
    Returns: dictionary containing measured optima and two scaling laws
    """
    optima = []
    selected_learning_rates = []
    selected_batch_sizes = []
    for scale_index in range(model_scales.numel()):
        flat_index = int(torch.argmin(measured_losses[scale_index].reshape(-1)))
        learning_rate_index = flat_index // batch_sizes.numel()
        batch_index = flat_index % batch_sizes.numel()
        learning_rate = float(learning_rates[learning_rate_index])
        batch_size = float(batch_sizes[batch_index])
        optima.append({
            "model_scale": float(model_scales[scale_index]),
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "loss": float(measured_losses[scale_index, learning_rate_index, batch_index]),
        })
        selected_learning_rates.append(learning_rate)
        selected_batch_sizes.append(batch_size)

    log_scales = torch.log(model_scales.double())
    design = torch.stack((torch.ones_like(log_scales), log_scales), dim=1)
    log_learning_rates = torch.log(torch.tensor(selected_learning_rates, dtype=torch.float64, device=model_scales.device))
    log_batch_sizes = torch.log(torch.tensor(selected_batch_sizes, dtype=torch.float64, device=model_scales.device))
    learning_rate_fit = torch.linalg.lstsq(design, log_learning_rates).solution
    batch_size_fit = torch.linalg.lstsq(design, log_batch_sizes).solution
    learning_rate_residuals = log_learning_rates - design @ learning_rate_fit
    batch_size_residuals = log_batch_sizes - design @ batch_size_fit
    learning_rate_coefficient = float(torch.exp(learning_rate_fit[0]))
    batch_size_coefficient = float(torch.exp(batch_size_fit[0]))
    learning_rate_exponent = float(learning_rate_fit[1])
    batch_size_exponent = float(batch_size_fit[1])
    return {
        "optima": optima,
        "learning_rate_coefficient": learning_rate_coefficient,
        "learning_rate_exponent": learning_rate_exponent,
        "batch_size_coefficient": batch_size_coefficient,
        "batch_size_exponent": batch_size_exponent,
        "learning_rate_residuals": learning_rate_residuals,
        "batch_size_residuals": batch_size_residuals,
        "target_learning_rate": float(learning_rate_coefficient * target_scale ** learning_rate_exponent),
        "target_batch_size": float(batch_size_coefficient * target_scale ** batch_size_exponent),
    }