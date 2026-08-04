def compute_monitoring_metrics(system_type, y_true, y_pred):
    """
    Compute the appropriate monitoring metrics for the given system type.
    """
    if system_type == "classification":
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        accuracy = (tp + tn) / len(y_true)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return sorted([("accuracy", accuracy), ("f1", f1), ("precision", precision), ("recall", recall)])
    elif system_type == "regression":
        n = len(y_true)
        mae = sum(abs(t - p) for t, p in zip(y_true, y_pred)) / n
        rmse = (sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / n) ** 0.5
        return sorted([("mae", mae), ("rmse", rmse)])
    elif system_type == "ranking":
        paired = sorted(zip(y_pred, y_true), reverse=True)
        top_3 = paired[:3]
        relevant_in_top = sum(1 for _, rel in top_3 if rel == 1)
        total_relevant = sum(1 for t in y_true if t == 1)
        p_at_3 = relevant_in_top / 3
        r_at_3 = relevant_in_top / total_relevant if total_relevant > 0 else 0.0
        return sorted([("precision_at_3", p_at_3), ("recall_at_3", r_at_3)])
