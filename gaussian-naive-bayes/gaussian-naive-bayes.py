def gaussian_naive_bayes(X_train, y_train, X_test):
    """
    Predict class labels for test samples using Gaussian Naive Bayes.
    """
    # Write code here
    classes = sorted(set(y_train))
    n, d = len(y_train), len(X_train[0])
    class_data = {c: [] for c in classes}
    for i in range(n):
        class_data[y_train[i]].append(X_train[i])
    priors, means, variances = {}, {}, {}
    for c in classes:
        data = class_data[c]
        nc = len(data)
        priors[c] = nc / n
        means[c] = [sum(data[i][j] for i in range(nc)) / nc for j in range(d)]
        variances[c] = [sum((data[i][j] - means[c][j]) ** 2 for i in range(nc)) / nc for j in range(d)]
    eps = 1e-9
    preds = []
    for x in X_test:
        best_c, best_lp = classes[0], float("-inf")
        for c in classes:
            lp = math.log(priors[c])
            for j in range(d):
                var = variances[c][j] + eps
                lp += -0.5 * math.log(2 * math.pi * var) - (x[j] - means[c][j]) ** 2 / (2 * var)
            if lp > best_lp:
                best_lp = lp
                best_c = c
        preds.append(best_c)
    return preds