import numpy as np
import math

def detect_skew(train_dist, serving_dist, threshold=0.2, eps=1e-10):
    """
    Detect train-serving skew using PSI.
    """
    results = {}
    for feature in sorted(train_dist.keys()):
        train = train_dist[feature]
        serve = serving_dist[feature]
        psi = 0.0
        for i in range(len(train)):
            t = train[i] + eps
            s = serve[i] + eps
            psi += (s - t) * math.log(s / t)
        psi = abs(psi)
        results[feature] = {"psi": psi, "skewed": psi >= threshold}
    return results
