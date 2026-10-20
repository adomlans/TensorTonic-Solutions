import math
import torch

def fit_data_power_law(data_sizes, losses, target_sizes, floor_candidates):
    """
    Returns: dictionary containing the selected fit and predictions
    """
    sizes = data_sizes.double()
    observed = losses.double()
    log_sizes = torch.log(sizes)
    design = torch.stack((torch.ones_like(log_sizes), log_sizes), dim=1)
    best = None

    for candidate in floor_candidates:
        floor = float(candidate)
        shifted = observed - floor
        if torch.any(shifted <= 0):
            continue
        intercept, slope = torch.linalg.lstsq(design, torch.log(shifted)).solution
        coefficient = float(torch.exp(intercept))
        exponent = float(-slope)
        if coefficient <= 0 or exponent <= 0 or not math.isfinite(coefficient + exponent):
            continue
        fitted = floor + coefficient * sizes.pow(-exponent)
        rmse = float(torch.sqrt(torch.mean((fitted - observed).square())))
        if best is None or rmse < best[0]:
            best = (rmse, floor, coefficient, exponent)

    rmse, floor, coefficient, exponent = best
    predictions = floor + coefficient * target_sizes.double().pow(-exponent)
    return {
        "floor": floor,
        "coefficient": coefficient,
        "exponent": exponent,
        "predictions": predictions,
        "rmse": rmse,
    }