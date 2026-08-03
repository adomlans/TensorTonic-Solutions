def calibrate_isotonic(cal_labels, cal_probs, new_probs):
    """
    Apply isotonic regression calibration.
    """
    pairs = sorted(zip(cal_probs, cal_labels))
    sorted_probs = [p for p, l in pairs]
    sorted_labels = [float(l) for p, l in pairs]
    blocks = []
    for val in sorted_labels:
        blocks.append([val, 1])
        while len(blocks) >= 2 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            blocks[-2][0] += blocks[-1][0]
            blocks[-2][1] += blocks[-1][1]
            blocks.pop()
    cal_values = []
    for val_sum, count in blocks:
        cal_values.extend([val_sum / count] * count)
    result = []
    for q in new_probs:
        if q <= sorted_probs[0]:
            result.append(cal_values[0])
        elif q >= sorted_probs[-1]:
            result.append(cal_values[-1])
        else:
            lo, hi = 0, len(sorted_probs) - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if sorted_probs[mid] <= q:
                    lo = mid
                else:
                    hi = mid - 1
            p1, v1 = sorted_probs[lo], cal_values[lo]
            p2, v2 = sorted_probs[lo + 1], cal_values[lo + 1]
            t = (q - p1) / (p2 - p1) if p2 != p1 else 0
            result.append(v1 + t * (v2 - v1))
    return result
