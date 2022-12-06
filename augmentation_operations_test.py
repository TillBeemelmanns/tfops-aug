"""Tests for augmentation_operations.py"""

import tensorflow as tf
import cv2

from augmentation_operations import int_parameter, float_parameter, AUGMENTATION_BY_NAME


class TestIntParameter(tf.test.TestCase):
    def test_int_parameter(self):
        for i in range(10):
            value = int_parameter(i, maxval=10, minval=0)
            expected = i
            self.assertEqual(value, expected)
            self.assertDTypeEqual(value, int)


class TestIntParameter2(tf.test.TestCase):
    def test_int_parameter(self):
        expected_values = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        for i in range(11):
            value = int_parameter(i, maxval=20, minval=10)
            expected = expected_values[i]
            self.assertEqual(value, expected)
            self.assertDTypeEqual(value, int)


class TestIntParameter3(tf.test.TestCase):
    def test_int_parameter(self):
        for maxval, minval in zip([5, 10, 80], [0, 0, 70]):
            for i in range(11):
                value = int_parameter(i, maxval=maxval, minval=minval)
                self.assertDTypeEqual(value, int)
                if i == 0:
                    self.assertEqual(value, minval)
                elif i == 10:
                    self.assertEqual(value, maxval)


class TestFloatParameter1(tf.test.TestCase):
    def test_float_parameter(self):
        for i in range(10):
            value = float_parameter(i, maxval=10, minval=0)
            expected = float(i)
            self.assertEqual(value, expected)
            self.assertDTypeEqual(value, float)


class TestFloatParameter2(tf.test.TestCase):
    def test_float_parameter(self):
        for maxval, minval in zip([5, 10, 80], [0, 0, 70]):
            value = float_parameter(0, maxval=maxval, minval=minval)
            self.assertEqual(value, minval)
            self.assertDTypeEqual(value, float)

            value = float_parameter(10, maxval=maxval, minval=minval)
            self.assertEqual(value, maxval)
            self.assertDTypeEqual(value, float)


class TestDtype(tf.test.TestCase):
    def test_dtype(self):
        img_org = cv2.cvtColor(cv2.imread("assets/test_image.jpg"), cv2.COLOR_BGR2RGB)

        for name, op in AUGMENTATION_BY_NAME.items():
            img = tf.convert_to_tensor(img_org)
            img = op(img, 5)
            self.assertDTypeEqual(img, tf.uint8)
            self.assertGreaterEqual(tf.reduce_max(img), 0)
            self.assertLessEqual(tf.reduce_min(img), 255)


if __name__ == '__main__':
    tf.test.main()
