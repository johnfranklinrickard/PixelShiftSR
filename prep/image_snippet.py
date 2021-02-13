import numpy
import os


def extract_random_snippet(image, snippet_x=256, snippet_y=256):
    assert image.shape[0] > snippet_x and image.shape[1] > snippet_y
    max_x = image.shape[0] - snippet_x
    max_y = image.shape[1] - snippet_y
    start_x = numpy.random.randint(0, max_x)
    start_y = numpy.random.randint(0, max_y)
    end_x = start_x + snippet_x
    end_y = start_y + snippet_y
    return image[start_x:end_x, start_y:end_y], start_x, start_y


def images_from_dir(directory_name=None):
    for entry in os.scandir(directory_name):
        if entry.name.endswith(".png") and entry.is_file():
            yield entry


def no_shift():
    return 0, 0


def half_shift():
    return 0, 0.5


def quarter_shift():
    return 0, 0.25, 0.5, 0.75


# hard coded random array for reproducability
def random0_shifts():
    return (0.45827556, 0.81700253, 0.10976018, 0.94226881, 0.92507905,
            0.3281929, 0.76806125, 0.71006357, 0.92269042, 0.03214076,
            0.11784005, 0.77842902, 0.60397358, 0.10120204, 0.2834368,
            0.14302333, 0.81090329, 0.14976199, 0.8049117, 0.84405617,
            0.25479715, 0.8276375, 0.71067842, 0.80527478, 0.76914734,
            0.02411469, 0.16829006, 0.24266901, 0.23532346, 0.82618664,
            0.87076253, 0.29077999, 0.53627999, 0.08863232, 0.17803077,
            0.04235153, 0.73004863, 0.12771995, 0.50726152, 0.28726838,
            0.76762234, 0.14399337, 0.9058409, 0.15567541, 0.93813085,
            0.97081135, 0.98268429, 0.69171163, 0.20419215, 0.57377654,
            0.97479026, 0.53501836, 0.71777861, 0.16413508, 0.37370553,
            0.11900185, 0.47896497, 0.75426357, 0.9888634, 0.54197847,
            0.41962811, 0.66753586, 0.0359308, 0.1575505)
