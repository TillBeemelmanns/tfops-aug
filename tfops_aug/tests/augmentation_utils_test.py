"""Tests for augmentation_utils.py"""

import tensorflow as tf
import numpy as np
import cv2

from tfops_aug.augmentation_operations import ALL_AUGMENTATION_NAMES_AS_LIST
from tfops_aug.augmentation_policies import test_policy
from tfops_aug.augmentation_utils import apply_augmentation_policy


class TestAllAugmentations(tf.test.TestCase):
    def test_all_augmentations(self):
        img_org = cv2.cvtColor(cv2.imread("assets/test_image.jpg"), cv2.COLOR_BGR2RGB)

        for op in ALL_AUGMENTATION_NAMES_AS_LIST:
            augmentation_policy = {}
            subpolicy = {}
            subpolicy['op0'] = [op, 1, 10]
            augmentation_policy['sub_policy0'] = subpolicy
            img = tf.convert_to_tensor(img_org)
            img = apply_augmentation_policy(img, augmentation_policy)

            self.assertDTypeEqual(img, tf.uint8)
            self.assertEqual(img.shape, img_org.shape)
            self.assertGreaterEqual(np.min(img), 0)
            self.assertLessEqual(np.max(img), 255)


class TestAugmentationPolicy(tf.test.TestCase):
    def test_augmentation_policy(self):
        img_org = cv2.cvtColor(cv2.imread("assets/test_image.jpg"), cv2.COLOR_BGR2RGB)
        img = tf.convert_to_tensor(img_org)
        img = apply_augmentation_policy(img, test_policy)
        img = img.numpy()

        self.assertEqual(img.shape, img_org.shape)
        self.assertGreaterEqual(np.min(img), 0)
        self.assertLessEqual(np.max(img), 255)


if __name__ == '__main__':
    tf.test.main()
