import numpy as np

def decision_tree_split(X, y):
    """
    Find the best feature and threshold to split the data.
    """
    # Write code here
    n, d = len(y), len(X[0])
    def gini(labels):
        t = len(labels)
        if t == 0:
            return 0.0
        counts = {}
        for l in labels:
            counts[l] = counts.get(l, 0) + 1
        return 1.0 - sum((c / t) ** 2 for c in counts.values())
    best_gain, best_feat, best_thresh = -1.0, 0, 0.0
    parent_gini = gini(y)
    for f in range(d):
        vals = sorted(set(X[i][f] for i in range(n)))
        for vi in range(len(vals) - 1):
            thresh = (vals[vi] + vals[vi + 1]) / 2.0
            ly = [y[i] for i in range(n) if X[i][f] <= thresh]
            ry = [y[i] for i in range(n) if X[i][f] > thresh]
            if not ly or not ry:
                continue
            wg = len(ly) / n * gini(ly) + len(ry) / n * gini(ry)
            gain = parent_gini - wg
            if gain > best_gain:
                best_gain, best_feat, best_thresh = gain, f, thresh
    return [best_feat, best_thresh]