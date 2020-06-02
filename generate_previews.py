import tensorflow as tf
from input_preprocess import *
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':

    augmentation_policy = {'sub_policy0': {'op0': ['add_noise', 1, 10]}}

    img = cv2.cvtColor(cv2.imread("assets/test_image.jpg"), cv2.COLOR_BGR2RGB)

    img = tf.convert_to_tensor(img)

    img = tf.cast(img, dtype=tf.uint8)

    img = apply_augmentation_policy(img, augmentation_policy)

    img = img.numpy()

    plt.imshow(img/255.0)
    plt.show()
