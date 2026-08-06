def linear_interpolation(values):
    """
    Fill missing (None) values using linear interpolation.
    """
    result = values[:]
    n = len(result)
    i = 0
    while i < n:
        if result[i] is None:
            left = i - 1
            right = i + 1
            while right < n and result[right] is None:
                right += 1
            left_val = result[left]
            right_val = result[right]
            span = right - left
            for j in range(left + 1, right):
                t = (j - left) / span
                result[j] = left_val + t * (right_val - left_val)
            i = right
        else:
            i += 1
    return result
