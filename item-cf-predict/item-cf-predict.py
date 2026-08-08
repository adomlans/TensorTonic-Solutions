def item_cf_predict(user_ratings, item_similarities, target):
    """
    Predict the rating using item-based collaborative filtering.
    """
    num = 0.0
    den = 0.0
    for i in range(len(user_ratings)):
        if i == target or user_ratings[i] == 0:
            continue
        sim = item_similarities[i]
        if sim > 0:
            num += sim * user_ratings[i]
            den += sim
    if den == 0:
        return 0.0
    return num / den
