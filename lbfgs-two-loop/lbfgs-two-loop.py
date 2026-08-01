def _dot(a, b):
    """Dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))

def lbfgs_direction(grad, s_list, y_list):
    """
    Compute the L-BFGS search direction using the two-loop recursion.
    """
    # Write code here
    n = len(grad)
    m = len(s_list)
    q = list(grad)
    alphas = [0.0] * m
    rhos = [0.0] * m
    for i in range(m - 1, -1, -1):
        rhos[i] = 1.0 / _dot(y_list[i], s_list[i])
        alphas[i] = rhos[i] * _dot(s_list[i], q)
        q = [q[j] - alphas[i] * y_list[i][j] for j in range(n)]
    gamma = _dot(s_list[-1], y_list[-1]) / _dot(y_list[-1], y_list[-1])
    r = [gamma * q[j] for j in range(n)]
    for i in range(m):
        beta = rhos[i] * _dot(y_list[i], r)
        r = [r[j] + s_list[i][j] * (alphas[i] - beta) for j in range(n)]
    return [-r[j] for j in range(n)]
