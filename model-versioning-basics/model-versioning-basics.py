def promote_model(models):
    """
    Decide which model version to promote to production.
    """
    best = None
    for m in models:
        if best is None:
            best = m
            continue
        if m["accuracy"] > best["accuracy"]:
            best = m
        elif m["accuracy"] == best["accuracy"]:
            if m["latency"] < best["latency"]:
                best = m
            elif m["latency"] == best["latency"]:
                if m["timestamp"] > best["timestamp"]:
                    best = m
    return best["name"]
