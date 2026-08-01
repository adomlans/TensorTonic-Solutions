import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):
    """
    Compute log-likelihood P(y|x) for Bernoulli Naive Bayes.
    """
    # Write code here
    classes = sorted(set(y_train))
    n = len(y_train)
    d = len(X_train[0])
    class_data = {c: [] for c in classes}
    for i in range(n):
        class_data[y_train[i]].append(X_train[i])
    priors = {}
    feat_probs = {}
    for c in classes:
        data = class_data[c]
        nc = len(data)
        priors[c] = nc / n
        feat_probs[c] = []
        for j in range(d):
            count_j = sum(1 for row in data if row[j] == 1)
            feat_probs[c].append((count_j + 1) / (nc + 2))
    result = []
    for x in X_test:
        log_posts = []
        for c in classes:
            lp = math.log(priors[c])
            for j in range(d):
                if x[j] == 1:
                    lp += math.log(feat_probs[c][j])
                else:
                    lp += math.log(1 - feat_probs[c][j])
            log_posts.append(round(lp, 4))
        result.append(log_posts)
    return result
    pass