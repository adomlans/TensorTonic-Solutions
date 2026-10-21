import math
import torch

def isoflop_optimal_frontier(compute_budgets, parameter_counts, terminal_losses):
    """
    Returns: dictionary containing optima and the fitted frontier
    """
    optima = []
    optimum_parameters = []
    for row in range(compute_budgets.numel()):
        sizes = parameter_counts[row].double()
        observed = terminal_losses[row].double()
        x = torch.log(sizes)
        design = torch.stack((x.square(), x, torch.ones_like(x)), dim=1)
        curvature, linear, intercept = [
            float(value) for value in torch.linalg.lstsq(design, observed).solution
        ]
        optimum_log_parameters = -linear / (2 * curvature)
        parameters = math.exp(optimum_log_parameters)
        compute = float(compute_budgets[row])
        tokens = compute / (6 * parameters)
        loss = (
            curvature * optimum_log_parameters ** 2
            + linear * optimum_log_parameters
            + intercept
        )
        optima.append({
            "compute": compute,
            "parameters": parameters,
            "tokens": tokens,
            "loss": loss,
        })
        optimum_parameters.append(parameters)

    log_compute = torch.log(compute_budgets.double())
    log_parameters = torch.log(
        torch.tensor(optimum_parameters, dtype=torch.float64, device=compute_budgets.device)
    )
    design = torch.stack((torch.ones_like(log_compute), log_compute), dim=1)
    intercept, exponent = torch.linalg.lstsq(design, log_parameters).solution
    return {
        "optima": optima,
        "frontier_scale": float(torch.exp(intercept)),
        "frontier_exponent": float(exponent),
    }
