import tensorflow as tf
import matplotlib.pyplot as plt
import time
import cv2
import pprint

from data_generator import Dataset
import common

from augmentation_operations import ALL_AUGMENTATION_NAMES_AS_LIST, apply_augmentation_policy
from augmentation_policies import augmentation_policy


def tf_dataset_pipeline():

    dataset = Dataset(
        dataset_name="test_dataset",
        split_name="train",
        dataset_dir=None,
        batch_size=4,
        crop_size=None,
        resize_factor=None,
        min_scale_factor=1.,
        max_scale_factor=1.,
        scale_factor_step_size=0,
        num_parallel_reads=1,
        num_parallel_calls=4,
        is_training=True,
        should_shuffle=True,
        should_repeat=False,
        augmentation_policy=augmentation_policy,
        num_samples=50,
        ignore_label=None
    )

    tf_dataset = dataset.get_tf_dataset()

    print(tf_dataset)

    plot = False
    epochs = 10

    start_time = time.perf_counter()
    for epoch in range(epochs):
        for element in tf_dataset:

            # Performing a training step
            time.sleep(0.01)

            if plot:
                image = element[common.IMAGE].numpy()[0, :, :, :]
                # cv2.imwrite("assets/input_image2.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                plt.imshow(image/255.0)
                plt.show()

    print("Augmented #{0} images".format(dataset.num_samples * epochs))
    print("Execution time:", time.perf_counter() - start_time)


def measure_each_augmentation_method(iterations_per_augmentation=100):
    """

    :param iterations_per_augmentation:
    :return:
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
            img = tf.cast(img, dtype=tf.float32)
            img = apply_augmentation_policy(img, augmentation_policy)
            _ = img.numpy()

        time_per_augmentation[op] = (time.perf_counter() - start_time) / iterations_per_augmentation

    pprint.pprint(time_per_augmentation)


if __name__ == '__main__':
    tf_dataset_pipeline()
    measure_each_augmentation_method()
