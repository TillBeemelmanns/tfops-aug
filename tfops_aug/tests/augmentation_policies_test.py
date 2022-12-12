"""Tests for augmentation_policies.py"""

import tensorflow as tf

from tfops_aug.augmentation_utils import validate_augmentation_policy
from tfops_aug.augmentation_policies import ALL_POLICIES


class ValidateAllAugmentationPolicies(tf.test.TestCase):
    def test_augmentation_policies(self):
        for policy in ALL_POLICIES:
            self.assertTrue(validate_augmentation_policy(policy))


if __name__ == '__main__':
    tf.test.main()
