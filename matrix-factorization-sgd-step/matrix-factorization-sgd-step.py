import numpy as np

def matrix_factorization_sgd_step(U, V, r, lr, reg):
    """
    Perform one SGD step for matrix factorization.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    e = r - float(U @ V)
    U_new = U + lr * (e * V - reg * U)
    V_new = V + lr * (e * U - reg * V)
    return [round(float(x), 4) for x in U_new], [round(float(x), 4) for x in V_new]
