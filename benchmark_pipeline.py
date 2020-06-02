import numpy as np
import matplotlib.pyplot as plt
import time
import cv2

from data_generator import Dataset
import common

if __name__ == '__main__':

    augmentation_policy = {'sub_policy0': {'op0': ['adjust_saturation', 0.2, 2],
                                           'op1': ['equalize', 0.1, 6],
                                           'op2': ['add_noise', 0.9, 6]},
                           'sub_policy1': {'op0': ['adjust_contrast', 0.1, 7],
                                           'op1': ['add_noise', 0.0, 10]},
                           'sub_policy2': {'op0': ['posterize', 0.9, 6],
                                           'op1': ['unbiased_gamma_sampling', 0.5, 1]},
                           'sub_policy3': {'op0': ['adjust_brightness', 0.3, 1],
                                           'op1': ['adjust_hue', 0.4, 5]},
                           'sub_policy4': {'op0': ['adjust_saturation', 0.2, 9],
                                           'op1': ['add_noise', 0.1, 0]},
                           'sub_policy5': {'op0': ['adjust_contrast', 1.0, 1],
                                           'op1': ['unbiased_gamma_sampling', 0.4, 9]},
                           'sub_policy6': {'op0': ['unbiased_gamma_sampling', 0.3, 0],
                                           'op1': ['adjust_hue', 0.1, 6]},
                           'sub_policy7': {'op0': ['solarize', 0.6, 0],
                                           'op1': ['adjust_gamma', 0.3, 6]},
                           'sub_policy8': {'op0': ['adjust_jpeg_quality', 0.7, 10],
                                           'op1': ['adjust_hue', 0.1, 2]},
                           'sub_policy9': {'op0': ['equalize', 0.6, 0],
                                           'op1': ['solarize', 0.0, 6]}}

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
        augmentation_policy=None,
        num_samples=50,
        ignore_label=None
    )

    tf_dataset = dataset.get_tf_dataset()

    print(tf_dataset)

    plot = True
    epochs = 1

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
