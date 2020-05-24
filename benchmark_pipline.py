import numpy as np
import matplotlib.pyplot as plt
import time

from data_generator import Dataset
import common


if __name__ == '__main__':

    augmentation_policy = {'sub_policy0': {'op1': ['adjust_saturation', 1, 2],
                                           'op2': ['equalize', 1, 6]},
                           'sub_policy1': {'op1': ['adjust_contrast', 1, 7],
                                           'op2': ['add_noise', 1, 10]},
                           'sub_policy2': {'op1': ['posterize', 1, 6],
                                           'op2': ['unbiased_gamma_sampling', 1, 1]},
                           'sub_policy3': {'op1': ['adjust_brightness', 1, 1],
                                           'op2': ['adjust_hue', 1, 5]},
                           'sub_policy4': {'op1': ['adjust_saturation', 1, 9],
                                           'op2': ['add_noise', 1, 0]},
                           'sub_policy5': {'op1': ['adjust_contrast', 1, 1],
                                           'op2': ['unbiased_gamma_sampling', 1, 9]},
                           'sub_policy6': {'op1': ['unbiased_gamma_sampling', 1, 0],
                                           'op2': ['adjust_hue', 1, 6]},
                           'sub_policy7': {'op1': ['solarize', 1, 0],
                                           'op2': ['adjust_gamma', 1, 6]},
                           'sub_policy8': {'op1': ['adjust_jpeg_quality', 1, 10],
                                           'op2': ['adjust_hue', 1, 2]},
                           'sub_policy9': {'op1': ['equalize', 1, 0],
                                           'op2': ['solarize', 1, 6]}}

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
        should_shuffle=False,
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
                plt.imshow(image/255.0)
                plt.show()

    print("Augmented #{0} images".format(dataset.num_samples * epochs))
    print("Execution time:", time.perf_counter() - start_time)
