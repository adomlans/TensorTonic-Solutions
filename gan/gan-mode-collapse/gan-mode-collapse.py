import numpy as np

def detect_mode_collapse(generated_samples, threshold=0.1):
    samples = np.array(generated_samples, dtype=float)
    std_devs = np.std(samples, axis=0)
    mean_std = np.mean(std_devs)
    is_collapsed = bool(mean_std < threshold)
    return {"diversity_score": round(float(mean_std), 4), "is_collapsed": is_collapsed}
