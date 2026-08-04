import math

def evaluate_shadow(production_log, shadow_log, criteria):
    """
    Evaluate whether a shadow model is ready for promotion.
    """
    n = len(production_log)
    prod_correct = sum(1 for p in production_log if p["prediction"] == p["actual"])
    shadow_correct = sum(1 for s in shadow_log if s["prediction"] == s["actual"])
    prod_accuracy = prod_correct / n
    shadow_accuracy = shadow_correct / n
    accuracy_gain = shadow_accuracy - prod_accuracy
    sorted_latencies = sorted(s["latency_ms"] for s in shadow_log)
    p95_index = math.ceil(0.95 * n) - 1
    shadow_p95 = sorted_latencies[p95_index]
    agreements = sum(1 for p, s in zip(production_log, shadow_log) if p["prediction"] == s["prediction"])
    agreement_rate = agreements / n
    promote = (
        accuracy_gain >= criteria["min_accuracy_gain"]
        and shadow_p95 <= criteria["max_latency_p95"]
        and agreement_rate >= criteria["min_agreement_rate"]
    )
    return {
        "promote": promote,
        "metrics": {
            "shadow_accuracy": shadow_accuracy,
            "production_accuracy": prod_accuracy,
            "accuracy_gain": accuracy_gain,
            "shadow_latency_p95": shadow_p95,
            "agreement_rate": agreement_rate,
        }
    }
