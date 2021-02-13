import unittest
import numpy
import torch
import tests.test_helper as h


def n_ones():
    return numpy.ones((3, 3, 3))


def n_zeros():
    return numpy.zeros((3, 3, 3))


def t_ones():
    return torch.ones((3, 3, 3))


def t_zeros():
    return torch.zeros((3, 3, 3))


class TestPSNR(unittest.TestCase):
    def test_mse_numpy(self):
        from utility.psnr import mse
        value = mse(n_ones() * 2, n_zeros())
        self.assertTrue(h.close_equal(value, 4.))

    def test_mse_tensor(self):
        from utility.psnr import mse
        value = mse(t_ones() * 0.25, t_zeros())
        self.assertTrue(h.close_equal(value, 0.0625))

    def test_PSNR_numpy(self):
        from utility.psnr import psnr
        value = psnr(n_ones() * 10, n_zeros(), max_value=255.)
        self.assertTrue(h.close_equal(value, 28.1308036086791))

    def test_PSNR_tensor(self):
        from utility.psnr import psnr
        value = psnr(t_ones() * 10, t_zeros(), max_value=255.)
        self.assertTrue(h.close_equal(value, 28.1308036086791))


if __name__ == '__main__':
    unittest.main()
