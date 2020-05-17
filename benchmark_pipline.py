import numpy as np
import matplotlib.pyplot as plt
import time

from data_generator import Dataset
import common


if __name__ == '__main__':

    augmentation_policy =  { 'sub_policy1': {'op1': ['adjust_saturation', 0.2, 2],
                                             'op2': ['equalize', 0.7, 6]},
                             'sub_policy2': {'op1': ['adjust_contrast', 0.9, 7],
                                             'op2': ['add_noise', 0.2, 10]},
                             'sub_policy3': {'op1': ['posterize', 0.8, 6],
                                             'op2': ['unbiased_gamma_sampling', 0.4, 1]},
                             'sub_policy4': {'op1': ['adjust_brightness', 0.3, 1],
                                             'op2': ['adjust_hue', 0.2, 5]},
                             'sub_policy5': {'op1': ['adjust_saturation', 0.5, 9],
                                             'op2': ['add_noise', 0.6, 0]},
                             'sub_policy6': {'op1': ['adjust_contrast', 0.0, 1],
                                             'op2': ['unbiased_gamma_sampling', 0.1, 9]},
                             'sub_policy7': {'op1': ['unbiased_gamma_sampling', 0.4, 0],
                                             'op2': ['adjust_hue', 0.5, 6]},
                             'sub_policy8': {'op1': ['solarize', 0.8, 0],
                                             'op2': ['adjust_gamma', 0.4, 6]},
                             'sub_policy9': {'op1': ['adjust_jpeg_quality', 0.2, 10],
                                             'op2': ['adjust_hue', 0.0, 2]},
                             'sub_policy10':{'op1': ['equalize', 0.0, 0],
                                             'op2': ['solarize', 0.0, 6]}}


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
        num_readers=4,
        is_training=True,
        should_shuffle=False,
        should_repeat=False,
        augmentation_policy=augmentation_policy,
        num_samples=1000,
        ignore_label=None
    )

    tf_dataset = dataset.get_tf_dataset()

    print(tf_dataset)

    plot = False

    start_time = time.perf_counter()
    for element in tf_dataset:

        # Performing a training step
        time.sleep(0.01)

        if plot:
            image = element[common.IMAGE].numpy()[0, :, :, :]
            plt.imshow(image/255.0)
            plt.show()

    print("Execution time:", time.perf_counter() - start_time)
