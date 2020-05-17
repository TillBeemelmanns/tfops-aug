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
        model_variant=None,
        num_readers=1,
        is_training=False,
        should_shuffle=False,
        should_repeat=False,
        augmentation_policy=None,
        num_samples=100,
        ignore_label=None
    )

    tf_dataset = dataset.get_tf_dataset()

    print(tf_dataset)

    for element in tf_dataset:
        image = element[common.IMAGE].numpy()
        print(image.shape)
