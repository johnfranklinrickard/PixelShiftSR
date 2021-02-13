import unittest


class TestMathUtil(unittest.TestCase):
    def test_clamp_index(self):
        from utility.math_util import clamp_index
        self.assertEqual(clamp_index(-1, 5), 0)
        self.assertEqual(clamp_index(5, 5), 4)
        self.assertEqual(clamp_index(2, 4), 2)

    def test_nearest_pixel_indices(self):
        from utility.math_util import nearest_pixel_indices
        x, y = nearest_pixel_indices(0., 4)
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        x, y = nearest_pixel_indices(5.6, 8)
        self.assertEqual(x, 5)
        self.assertEqual(y, 6)
        x, y = nearest_pixel_indices(31.5, 32)
        self.assertEqual(x, 31)
        self.assertEqual(y, 31)

    def test_float_to_int_remainder(self):
        from utility.math_util import float_to_int_remainder
        actual = float_to_int_remainder(1.67)
        expected = 0.67
        self.assertAlmostEqual(actual, expected)

    def test_interpolate(self):
        from utility.math_util import interpolate
        actual = interpolate(3, 1, 0)
        expected = 3
        self.assertAlmostEqual(actual, expected)
        actual = interpolate(5, 8, 1)
        expected = 8
        self.assertAlmostEqual(actual, expected)
        actual = interpolate(0, 10, 0.5)
        expected = 5
        self.assertAlmostEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
