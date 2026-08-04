def feature_store_lookup(feature_store, requests, defaults):
    """
    Join offline user features with online request-time features.
    """
    results = []
    for req in requests:
        user_id = req["user_id"]
        offline = feature_store.get(user_id, defaults)
        combined = {}
        combined.update(offline)
        combined.update(req["online_features"])
        results.append(combined)
    return results
