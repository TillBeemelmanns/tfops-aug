import tensorflow as tf
import os
from input_preprocess import *
import cv2

if __name__ == '__main__':
    img_org = cv2.cvtColor(cv2.imread("assets/test_image.jpg"), cv2.COLOR_BGR2RGB)

    for op in ALL_AUGMENTATION_NAMES_AS_LIST:
        augmentation_policy = {}
        subpolicy = {}

        for level in range(11):
            subpolicy['op0'] = [op, 1.0, level]
            augmentation_policy['sub_policy0'] = subpolicy
            img = tf.convert_to_tensor(img_org)
            img = tf.cast(img, dtype=tf.uint8)
            img = apply_augmentation_policy(img, augmentation_policy)
            img = img.numpy().astype(dtype='uint8')
            img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            cv2.imwrite(f"output/{op}_{level+1}.jpg", img)

        os.system(f"convert $(ls -1 output/{op}_*.jpg | sort -V) output/{op}.gif")
        os.system(f"rm output/{op}_*.jpg")


