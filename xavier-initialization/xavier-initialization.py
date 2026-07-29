def xavier_initialization(W, fan_in, fan_out):
    """
    Scale raw weights to Xavier uniform initialization.
    """
    # Write code here
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return [[round(W[i][j] * 2 * limit - limit, 4) for j in range(len(W[0]))] for i in range(len(W))]