from PIL import Image
import numpy
import os
import torch

DOWNSCALE_FACTOR = 4


def save_array_image(array, name):
    # Simply saves the given array as color image
    rounded = numpy.rint(array)
    image = Image.fromarray(rounded.astype('uint8'))
    image.save(name)
    return


def init_downscale_image(image, downscale_factor=DOWNSCALE_FACTOR):
    # Initializes ndarray with zeros of downscaled size
    assert valid_downsizeable_color_image(image, downscale_factor)
    scaled_x = image.shape[0] // downscale_factor
    scaled_y = image.shape[1] // downscale_factor

    if torch.is_tensor(image):
        return torch.zeros((scaled_x, scaled_y, 3), None, image.dtype)
    else:
        return numpy.zeros((scaled_x, scaled_y, 3))


def values_are_in_range(array, minimum, maximum):
    for i in array:
        if minimum > i or i > maximum:
            return False

    return True


def valid_downsizeable_color_image(image, downscale_factor=DOWNSCALE_FACTOR):
    if len(image.shape) != 3:
        return False
    valid_x = image.shape[0] % downscale_factor == 0
    valid_y = image.shape[1] % downscale_factor == 0
    valid_z = image.shape[2] == 3
    return valid_x and valid_y and valid_z


def create_dir(name):
    if not os.path.exists(name):
        os.makedirs(name)


def load_img_ndarray(name):
    img = Image.open(name)
    return numpy.array(img)


def create_random_img_array_uint8(size_x, size_y, size_z=3):
    return numpy.random.randint(0, 256, (size_x, size_y, size_z))


def append_to_file(file_name, append_string):
    with open(file_name, "a") as file:
        file.write(append_string)


def create_file(file_name, optional_text=""):
    with open(file_name, "w") as file:
        file.write(optional_text)
