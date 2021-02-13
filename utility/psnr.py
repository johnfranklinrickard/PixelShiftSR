import torch
import numpy
from math import log10


# mean squared error
def mse(control, test):
    if torch.is_tensor(control) and torch.is_tensor(test):
        return torch.sum((control - test) ** 2) / control.numel()
    else:
        return numpy.sum((control - test) ** 2) / control.size


# https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
def psnr(control, test, max_value=255.):
    return 10 * log10(max_value ** 2 / mse(control, test))
