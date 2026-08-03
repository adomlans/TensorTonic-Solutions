import numpy as np

def mean_average_precision(y_true_list, y_score_list, k=None):
    """
    Compute Mean Average Precision (mAP) for multiple retrieval queries.
    """
    if len(y_true_list) != len(y_score_list):
        raise ValueError("y_true_list and y_score_list must have same length")
    ap_scores = []
    for y_true, y_score in zip(y_true_list, y_score_list):
        y_true = np.asarray(y_true, dtype=int)
        y_score = np.asarray(y_score, dtype=float)
        if len(y_true) != len(y_score):
            raise ValueError("length mismatch within query")
        num_relevant = int(np.sum(y_true))
        if num_relevant == 0:
            ap_scores.append(0.0)
            continue
        sorted_indices = np.argsort(-y_score)
        y_true_sorted = y_true[sorted_indices]
        n = len(y_true_sorted) if k is None else min(k, len(y_true_sorted))
        ap = 0.0
        hits = 0
        for i in range(n):
            if y_true_sorted[i] == 1:
                hits += 1
                ap += hits / (i + 1)
        ap /= num_relevant
        ap_scores.append(ap)
    map_score = float(np.mean(ap_scores)) if ap_scores else 0.0
    return round(map_score, 6), [round(a, 6) for a in ap_scores]
