import numpy as np
import cv2
from augmentation_operations import *

import unittest


class TestDerivativeMethods(unittest.TestCase):

    def test_all_augmentations(self):
        img_org = cv2.cvtColor(cv2.imread("assets/test_image.jpg"), cv2.COLOR_BGR2RGB)

        for op in ALL_AUGMENTATION_NAMES_AS_LIST:
            augmentation_policy = {}
            subpolicy = {}
            subpolicy['op0'] = [op, 1, 10]
            augmentation_policy['sub_policy0'] = subpolicy
            img = tf.convert_to_tensor(img_org)
            img = tf.cast(img, dtype=tf.uint8)
            img = apply_augmentation_policy(img, augmentation_policy)
            img = img.numpy()

            self.assertEqual(img.shape, img_org.shape)
            self.assertGreaterEqual(np.min(img), 0)
            self.assertLessEqual(np.max(img), 255)


    def test_augmentation_policy(self):
        policy = {'sub_policy0': {'op0': ['adjust_saturation', 1.0, 2],
                                  'op1': ['equalize', 1.0, 6],
                                  'op2': ['add_noise', 1.0, 6]},
                  'sub_policy1': {'op0': ['adjust_contrast', 1.0, 7],
                                  'op1': ['add_noise', 1.0, 10]},
                  'sub_policy2': {'op0': ['posterize', 1.0, 6],
                                  'op1': ['unbiased_gamma_sampling', 1.0, 1]},
                  'sub_policy3': {'op0': ['adjust_brightness', 1.0, 1],
                                  'op1': ['adjust_hue', 1.0, 5]},
                  'sub_policy4': {'op0': ['adjust_saturation', 0.2, 9],
                                  'op1': ['add_noise', 1.0, 0]},
                  'sub_policy5': {'op0': ['adjust_contrast', 1.0, 1],
                                  'op1': ['unbiased_gamma_sampling', 1.0, 9]},
                  'sub_policy6': {'op0': ['unbiased_gamma_sampling', 1.0, 0],
                                  'op1': ['adjust_hue', 1.0, 6]},
                  'sub_policy7': {'op0': ['solarize', 1.0, 0],
                                  'op1': ['adjust_gamma', 1.0, 6]},
                  'sub_policy8': {'op0': ['adjust_jpeg_quality', 1.0, 10],
                                  'op1': ['adjust_hue', 1.0, 2]},
                  'sub_policy9': {'op0': ['equalize', 1.0, 0],
                                  'op1': ['solarize', 1.0, 6]}}

        img_org = cv2.cvtColor(cv2.imread("assets/test_image.jpg"), cv2.COLOR_BGR2RGB)
        img = tf.convert_to_tensor(img_org)
        img = tf.cast(img, dtype=tf.uint8)
        img = apply_augmentation_policy(img, policy)
        img = img.numpy()

        self.assertEqual(img.shape, img_org.shape)
        self.assertGreaterEqual(np.min(img), 0)
        self.assertLessEqual(np.max(img), 255)

if __name__ == '__main__':
    unittest.main()
