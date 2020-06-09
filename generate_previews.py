import tensorflow as tf
import os
from augmentation_operations import *
from augmentation_policies import augmentation_policy
import cv2


def generate_all_augmentation_gifs(image):
    for op in ALL_AUGMENTATION_NAMES_AS_LIST:
        augmentation_policy = {}
        subpolicy = {}

        for level in range(11):
            subpolicy['op0'] = [op, 1.0, level]
            augmentation_policy['sub_policy0'] = subpolicy
            img = tf.convert_to_tensor(image)
            img = tf.cast(img, dtype=tf.uint8)
            img = apply_augmentation_policy(img, augmentation_policy)
            img = img.numpy().astype(dtype='uint8')
            img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"assets/{op}_{level}.jpg", img)

        os.system(f"convert $(ls -1 assets/{op}_*.jpg | sort -V) assets/{op}.gif")
        os.system(f"rm assets/{op}_*.jpg")


def generate_augmentation_policy_gif(image, policy):
    for i in range(20):
        img = apply_augmentation_policy(image, policy)
        img = img.numpy().astype(dtype='uint8')
        img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"assets/augmentation_policy_{i}.jpg", img)

    os.system(f"convert $(ls -1 assets/augmentation_policy_*.jpg | sort -V) assets/augmentation_policy.gif")
    os.system(f"rm assets/augmentation_policy_*.jpg")


if __name__ == '__main__':
    img_org = cv2.cvtColor(cv2.imread("assets/test_image.jpg"), cv2.COLOR_BGR2RGB)
    generate_augmentation_policy_gif(img_org, augmentation_policy)
    generate_all_augmentation_gifs(img_org)


