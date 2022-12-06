import tensorflow as tf
import matplotlib.pyplot as plt
import time
import cv2
import pprint

import common
from augmentation_operations import ALL_AUGMENTATION_NAMES_AS_LIST
from augmentation_policies import augmentation_policy
from augmentation_utils import apply_augmentation_policy


def parse_sample(image_path, rescale=False):
    """
    Argument:
    image_path -- String which contains the path to the camera image

    Returns:
    image_rgb -- tf.Tensor of size [1024, 2048, 3] as tf.uint8 containing the camera image
    """
    image_rgb = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)
    image_rgb.set_shape(common.IMAGE_SHAPE)

    if rescale:
        image_rgb = tf.image.resize(image_rgb, [1024, 2048], method=tf.image.ResizeMethod.BILINEAR)

    return image_rgb


def augmentor_func(img):
    img = apply_augmentation_policy(img, augmentation_policy)
    return img


def benachmark_dataset_pipeline(dataset_size=50, epochs=10, plot=False):
    dataset = tf.data.Dataset.from_tensor_slices([common.TEST_IMAGE_PATH])
    dataset = dataset.repeat(dataset_size)
    dataset = dataset.map(parse_sample, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(augmentor_func, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    start_time = time.perf_counter()
    for epoch in range(epochs):
        for augmented_image in dataset:

            # Performing a training step
            time.sleep(0.01)

            if plot:
                plt.imshow(augmented_image.numpy() / 255.)
                plt.show()

    print("Augmented {0} images".format(dataset_size * epochs))
    print("Execution time:", time.perf_counter() - start_time)


def measure_each_augmentation_method(iterations_per_augmentation=100):
    """
    Measure the time for each augmentation function. Executes each augmentation function <<iterations_per_augmentation>>
    times and computes the average.
    :param iterations_per_augmentation: Number of iterations per augmentation function
    :return: dictionary with augmentation function as key and time needed for one cycle as value in [s]
    """
    img_org = cv2.cvtColor(cv2.imread(common.TEST_IMAGE_PATH), cv2.COLOR_BGR2RGB)

    time_per_augmentation = {}

    for op in ALL_AUGMENTATION_NAMES_AS_LIST:
        start_time = time.perf_counter()
        for _ in range(iterations_per_augmentation):
            augmentation_policy = {}
            subpolicy = {}
            subpolicy['op0'] = [op, 1, 5]
            augmentation_policy['sub_policy0'] = subpolicy
            img = tf.convert_to_tensor(img_org)
            img = apply_augmentation_policy(img, augmentation_policy)
            _ = img.numpy()

        time_per_augmentation[op] = (time.perf_counter() - start_time) / iterations_per_augmentation
    return time_per_augmentation


if __name__ == '__main__':
    benachmark_dataset_pipeline(dataset_size=50, epochs=10, plot=False)
    result = measure_each_augmentation_method(iterations_per_augmentation=100)
    pprint.pprint(result)
