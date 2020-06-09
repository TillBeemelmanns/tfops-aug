import numpy as np
import matplotlib.pyplot as plt
import time
import cv2

from augmentation_policies import augmentation_policy
from data_generator import Dataset
import common

if __name__ == '__main__':

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
