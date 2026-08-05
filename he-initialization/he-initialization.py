import math

def he_initialization(W, fan_in):
    """
    Scale raw weights to He uniform initialization.
    """
    limit = math.sqrt(6.0 / fan_in)
    return [[round(W[i][j] * 2 * limit - limit, 4)
             for j in range(len(W[0]))] for i in range(len(W))]
