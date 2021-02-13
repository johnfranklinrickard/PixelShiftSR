import math


def clamp_index(value, index_count):
    if value < 0:
        return 0
    elif value > index_count - 1:
        return index_count - 1
    else:
        return value


def nearest_pixel_indices(index, max_index):
    # Returns the two nearest whole numbered indices to the given number
    # Remark: can return the same index as lower and upper for edge cases
    lower_index = clamp_index(math.floor(index), max_index)
    upper_index = clamp_index(math.ceil(index), max_index)
    return lower_index, upper_index


def float_to_int_remainder(value):
    # Returns the remainder after the dot of the floating point number
    assert value >= 0, "The function behaves different with negative values"
    return value - math.floor(value)


def interpolate(value1, value2, alpha):
    # Interpolates the given values according to alpha
    # alpha = 0. returns the first value, alpha = 1. returns the second
    return (1.0 - alpha) * value1 + alpha * value2
