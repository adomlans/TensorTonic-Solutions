import math
import torch

def data_mixture_extrapolation(mixture_names, data_sizes, losses,
                               target_size, floor_candidates):
    """
    Returns: dictionary containing all fits and the winning mixture
    """
    fits = []
    predictions = []
    for row, name in enumerate(mixture_names):
        sizes = data_sizes[row].double()
        observed = losses[row].double()
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
        prediction = float(floor + coefficient * target_size ** (-exponent))
        fits.append({"name": name, "floor": floor, "coefficient": coefficient, "exponent": exponent, "rmse": rmse})
        predictions.append(prediction)

    winner = min(range(len(predictions)), key=lambda index: (predictions[index], index))
    return {
        "fits": fits,
        "target_predictions": torch.tensor(predictions, dtype=torch.float64, device=data_sizes.device),
        "winning_mixture": mixture_names[winner],
        "winning_index": winner,
    }