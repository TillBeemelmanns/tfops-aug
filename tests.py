import tensorflow as tf
import numpy as np
import cv2
from input_preprocess import *

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


if __name__ == '__main__':
    unittest.main()
