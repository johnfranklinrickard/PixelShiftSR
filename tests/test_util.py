import unittest
import numpy


class TestUtil(unittest.TestCase):
    def test_init_downscale_image(self):
        from utility.util import init_downscale_image
        image = numpy.zeros((16, 64, 3))
        output = init_downscale_image(image)
        self.assertEqual(output.shape[0], 4)
        self.assertEqual(output.shape[1], 16)

    def test_values_are_in_range(self):
        from utility.util import values_are_in_range
        array = (0., 1.)
        self.assertTrue(values_are_in_range(array, 0., 1.))
        self.assertFalse(values_are_in_range(array, 0., 0.5))

    def test_valid_downsizeable_image_size(self):
        from utility.util import valid_downsizeable_color_image
        valid = numpy.zeros((32, 32, 3))
        invalid0 = numpy.zeros((16, 16))
        invalid1 = numpy.zeros((31, 32, 3))
        invalid2 = numpy.zeros((32, 31, 3))
        invalid3 = numpy.zeros((32, 32, 2))
        self.assertTrue(valid_downsizeable_color_image(valid))
        self.assertFalse(valid_downsizeable_color_image(invalid0))
        self.assertFalse(valid_downsizeable_color_image(invalid1))
        self.assertFalse(valid_downsizeable_color_image(invalid2))
        self.assertFalse(valid_downsizeable_color_image(invalid3))


if __name__ == '__main__':
    unittest.main()
