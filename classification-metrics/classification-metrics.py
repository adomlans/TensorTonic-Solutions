import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    """
    Compute accuracy, precision, recall, F1 for single-label classification.
    Averages: 'micro' | 'macro' | 'weighted' | 'binary' (uses pos_label).
    Return dict with float values.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    acc = float(np.mean(yt == yp)) if yt.size else 0.0
    labels = np.unique(np.concatenate([yt, yp]))
    K = labels.size
    idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(yt, yp):
        cm[idx[t], idx[p]] += 1
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec_c = tp / np.maximum(tp + fp, 1e-12)
    rec_c = tp / np.maximum(tp + fn, 1e-12)
    f1_c = 2 * prec_c * rec_c / np.maximum(prec_c + rec_c, 1e-12)
    support = cm.sum(axis=1).astype(float)
    if average == "macro":
        precision = float(np.mean(prec_c))
        recall = float(np.mean(rec_c))
        f1 = float(np.mean(f1_c))
    elif average == "weighted":
        w = support / np.maximum(support.sum(), 1e-12)
        precision = float((w * prec_c).sum())
        recall = float((w * rec_c).sum())
        f1 = float((w * f1_c).sum())
    elif average == "binary":
        i = idx.get(pos_label, -1)
        if i < 0:
            precision = recall = f1 = 0.0
        else:
            precision, recall, f1 = float(prec_c[i]), float(rec_c[i]), float(f1_c[i])
    else:
        TP = float(tp.sum())
        FP = float(fp.sum())
        FN = float(fn.sum())
        precision = TP / max(TP + FP, 1e-12)
        recall = TP / max(TP + FN, 1e-12)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"accuracy": round(acc, 6), "precision": round(precision, 6),
            "recall": round(recall, 6), "f1": round(f1, 6)}
