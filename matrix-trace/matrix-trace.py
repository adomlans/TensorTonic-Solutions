import numpy as np



def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")
    n = A.shape[0]
    s = 0
    for i in range(n):
        s += A[i, i]
    return float(s)

