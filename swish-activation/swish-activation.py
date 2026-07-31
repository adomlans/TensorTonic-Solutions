import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    # Write code here
    x=np.array(x,dtype=float)
    sigmoid=1.0/(1.0+np.exp(-x))
    result=x*sigmoid
    return np.round(result,4).tolist()
    pass