import os
import cv2

import tensorflow as tf

from tfops_aug.augmentation_operations import ALL_AUGMENTATION_NAMES_AS_LIST
from tfops_aug.augmentation_utils import apply_augmentation_policy
from tfops_aug.augmentation_policies import augmentation_policy

import config

def generate_all_augmentation_gifs(image, save_dir):
    for op in ALL_AUGMENTATION_NAMES_AS_LIST:
        augmentation_policy = {}
        subpolicy = {}

        for level in range(11):
            subpolicy['op0'] = [op, 1.0, level]
            augmentation_policy['sub_policy0'] = subpolicy
            img = tf.convert_to_tensor(image)
            img = apply_augmentation_policy(img, augmentation_policy)
            img = img.numpy()
            img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{save_dir}/{op}_{level}.jpg", img)

        os.system(f"convert $(ls -1 {save_dir}/{op}_*.jpg | sort -V) {save_dir}/{op}.gif")
        os.system(f"rm {save_dir}/{op}_*.jpg")


def generate_augmentation_policy_gif(image, policy, save_dir):
    for i in range(20):
        img = tf.convert_to_tensor(image)
        img = apply_augmentation_policy(img, policy)
        img = img.numpy()
        img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{save_dir}/augmentation_policy_{i}.jpg", img)

    os.system(f"convert $(ls -1 {save_dir}/augmentation_policy_*.jpg | sort -V) {save_dir}/augmentation_policy.gif")
    os.system(f"rm {save_dir}/augmentation_policy_*.jpg")


if __name__ == '__main__':
    img_org = cv2.cvtColor(cv2.imread(config.TEST_IMAGE_PATH), cv2.COLOR_BGR2RGB)
    generate_all_augmentation_gifs(img_org, config.SAVE_DIR)
    generate_augmentation_policy_gif(img_org, augmentation_policy, config.SAVE_DIR)
