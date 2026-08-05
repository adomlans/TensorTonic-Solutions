import numpy as np

def rotate_around_z(points, theta):
    """
    Rotate 3D point(s) around the Z-axis by angle theta (radians).
    """
    points = np.asarray(points, dtype=float)
    single = (points.ndim == 1)
    if single:
        points = points.reshape(1, 3)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    x_new = x * cos_t - y * sin_t
    y_new = x * sin_t + y * cos_t
    result = np.column_stack([x_new, y_new, z])
    if single:
        return result[0]
    return result
