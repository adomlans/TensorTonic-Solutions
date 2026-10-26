import numpy as np

def gradient_descent_step(values, gradients, learning_rate):
    """
    Returns: updated values and the predicted first-order objective change
    """
    values_array = np.asarray(values, dtype=np.float64)
    gradients_array = np.asarray(gradients, dtype=np.float64)
    rate = float(learning_rate)
    updated = values_array - rate * gradients_array
    predicted_change = float(np.dot(gradients_array, updated - values_array))
    return ([float(value) for value in updated], predicted_change)