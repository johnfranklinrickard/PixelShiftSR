import unittest
import numpy

import tests.test_helper as h
import prep.downscale as ds


class TestDownscale(unittest.TestCase):
    def test_uniform_img_shift(self):
        output = ds.batch_shift_downscale(h.zero_image(8, 8), h.shift_half())
        self.assertTrue(numpy.array_equal(output, h.zero_image(2, 2)))

    def test_uniform_img_shift2(self):
        output = ds.batch_shift_downscale(h.one_image(8, 8), (0, 0.33, 0.66))
        self.assertTrue(numpy.array_equal(output, h.one_image(2, 2)))

    def test_no_shift(self):
        output = ds.batch_shift_downscale(h.test_img0(), (0, 0))
        self.assertTrue(numpy.array_equal(output, h.downscaled_img0()))

    def test_no_shift2(self):
        output = ds.batch_shift_downscale(h.test_img1(), (0, 0))
        self.assertTrue(numpy.array_equal(output, h.downscaled_img1()))

    def test_shift_half(self):
        output = ds.batch_shift_downscale(h.test_img0(), h.shift_half())
        self.assertTrue(numpy.array_equal(output, h.downscaled_img0_shift_half()))

    def test_shift_half2(self):
        output = ds.batch_shift_downscale(h.test_img1(), h.shift_half())
        self.assertTrue(numpy.array_equal(output, h.downscaled_img1_shift_half()))

    def test_shift_thirds(self):
        output = ds.batch_shift_downscale(h.test_img0(), h.shift_thirds())
        self.assertTrue(h.close_equal(output, h.downscaled_img0_shift_thirds()))

    def test_shift_thirds2(self):
        output = ds.batch_shift_downscale(h.test_img1(), h.shift_thirds())
        self.assertTrue(h.close_equal(output, h.downscaled_img1_shift_thirds()))


if __name__ == '__main__':
    unittest.main()
