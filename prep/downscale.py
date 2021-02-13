from utility.util import *
from utility.math_util import *
import math


def batch_shift_downscale(image, row_pixel_shifts, downscale_factor=DOWNSCALE_FACTOR):
    # downscales the given image and shifts the rows according to
    # the given shift array(repeated of course)

    assert values_are_in_range(row_pixel_shifts, 0., 0.999)
    sub_image = init_downscale_image(image, downscale_factor)

    index_count_x = image.shape[0]  # number of x iterations
    shift_length = len(row_pixel_shifts)  # number of different shifts

    # create range with steps for y, as access is batched
    index_count_y = image.shape[1]
    y_indices = range(0, index_count_y, downscale_factor)

    for x in range(index_count_x):
        sx = x // downscale_factor  # x coordinate of the downscaled image
        pixel_shift = row_pixel_shifts[sx % shift_length] * downscale_factor
        for y in y_indices:
            shifted_y = y + pixel_shift
            # assumes downscale_factor == 4 currently
            sub_value = batch_sum_4_pixel(image, x, shifted_y, index_count_y)

            sy = y // downscale_factor  # y coordinate of the downscaled image
            sub_image[sx, sy] = sub_image[sx, sy] + sub_value

    return sub_image / math.pow(downscale_factor, 2)  # normalize pixels


def batch_sum_4_pixel(image, x, float_y, index_count):
    # batch adds aligned pixel fields and interpolates outer ones.
    # this removes interpolations of inner pixels, where the weights
    # add up to 1 anyway.
    alpha = float_to_int_remainder(float_y)
    start_y = math.floor(float_y)
    i0, i1, i2, i3, i4 = clamp_batch4_index(start_y, index_count)
    value = image[x, i0] * (1. - alpha)
    value = value + image[x, i1]
    value = value + image[x, i2]
    value = value + image[x, i3]
    value = value + image[x, i4] * alpha

    return value


def clamp_batch4_index(index, index_count):
    # batches the clamp index calculation at edges
    assert index >= 0

    i0 = min(index + 0, index_count - 1)
    i1 = min(index + 1, index_count - 1)
    i2 = min(index + 2, index_count - 1)
    i3 = min(index + 3, index_count - 1)
    i4 = min(index + 4, index_count - 1)
    return i0, i1, i2, i3, i4
