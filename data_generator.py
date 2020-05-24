import os

import collections
import glob
import tensorflow as tf
import pprint

import input_preprocess
import common

# Named tuple to describe the dataset properties.
DatasetDescriptor = collections.namedtuple(
    'DatasetDescriptor',
    [
        'splits_to_sizes',  # Splits of the dataset into training, val and test.
        'num_classes',  # Number of classes
        'ignore_label',  # Ignore label value.
        'local_dir',
        'class_names',
        'min_resize_value',
        'max_resize_value'
    ])

_TEST_DATASET = DatasetDescriptor(
    splits_to_sizes={
        'train': 2975,
        'val': 500,
    },
    ignore_label=255,
    num_classes=19,
    class_names=['Road', 'Sidewalk', 'Building', 'Wall', 'Fence', 'Pole', 'Traffic light',
                 'Traffic sign', 'Vegetation', 'Terrain', 'Sky', 'Person', 'Rider', 'Car',
                 'Truck', 'Bus', 'Train', 'Motorcycle', 'Bicycle'],
    local_dir="/home/beemelmanns/Documents/github/auto-augment-tf2-operations/data",
    min_resize_value=1024,
    max_resize_value=2048
)

_DATASETS_INFORMATION = {
    'test_dataset': _TEST_DATASET,
}

class Dataset:
    """Represents input dataset for deeplab model."""

    def __init__(self,
                 dataset_name,
                 split_name,
                 dataset_dir,
                 batch_size,
                 crop_size=None,
                 resize_factor=None,
                 min_scale_factor=1.,
                 max_scale_factor=1.,
                 scale_factor_step_size=0,
                 num_parallel_reads=1,
                 num_parallel_calls=4,
                 is_training=False,
                 should_shuffle=False,
                 shuffle_buffer_size=1,
                 should_repeat=False,
                 augmentation_policy=None,
                 num_samples=100,
                 ignore_label=None):
        """
        Initialize Dataset
        :param dataset_name:
        :param split_name:
        :param dataset_dir:
        :param batch_size:
        :param crop_size:
        :param resize_factor:
        :param min_scale_factor:
        :param max_scale_factor:
        :param scale_factor_step_size:
        :param num_parallel_reads:
        :param is_training:
        :param should_shuffle:
        :param should_repeat:
        :param augmentation_policy:
        :param ignore_label:
        """

        if dataset_name not in _DATASETS_INFORMATION:
            raise ValueError('The specified dataset "{0}" is not supported yet.'.format(dataset_name))
        self.dataset_name = dataset_name

        splits_to_sizes = _DATASETS_INFORMATION[dataset_name].splits_to_sizes

        if split_name not in splits_to_sizes:
            raise ValueError('data split name %s not recognized' % split_name)

        self.split_name = split_name

        self.batch_size = batch_size
        self.min_resize_value = _DATASETS_INFORMATION[dataset_name].min_resize_value
        self.max_resize_value = _DATASETS_INFORMATION[dataset_name].max_resize_value
        self.resize_factor = resize_factor
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.scale_factor_step_size = scale_factor_step_size
        self.num_parallel_reads = num_parallel_reads
        self.num_parallel_calls = num_parallel_calls
        self.is_training = is_training
        self.should_shuffle = should_shuffle
        self.should_repeat = should_repeat

        self.shuffle_buffer_size = shuffle_buffer_size

        # if crop_size is set use it
        if crop_size:
            self.crop_size = crop_size
        else:
            self.crop_size = _DATASETS_INFORMATION[dataset_name].min_resize_value + 1, \
                             _DATASETS_INFORMATION[dataset_name].max_resize_value + 1

        self.num_of_classes = _DATASETS_INFORMATION[self.dataset_name].num_classes
        self.class_names = _DATASETS_INFORMATION[self.dataset_name].class_names

        # override default path
        if dataset_dir is not None:
            self.dataset_dir = dataset_dir
        else:
            if os.environ.get("inside_docker"):
                raise NotImplementedError
            else:
                self.dataset_dir = _DATASETS_INFORMATION[dataset_name].local_dir

        if ignore_label:
            self.ignore_label = ignore_label
        else:
            self.ignore_label = _DATASETS_INFORMATION[self.dataset_name].ignore_label

        self.splits_to_sizes = _DATASETS_INFORMATION[self.dataset_name].splits_to_sizes

        self.num_samples = min(splits_to_sizes[self.split_name], num_samples)

        self.augmentation_policy = augmentation_policy

    def _parse_function(self, example_proto):
        """Function to parse the example proto.

        Args:
          example_proto: Proto in the format of tf.Example.

        Returns:
          A dictionary with parsed image, label, height, width and image name.

        Raises:
          ValueError: Label is of wrong shape.
        """

        # Currently only supports jpeg and png.
        # Need to use this logic because the shape is not known for
        # tf.image.decode_image and we rely on this info to
        # extend label if necessary.
        def _decode_image(content, channels):
            return tf.cond(
                tf.image.is_jpeg(content),
                lambda: tf.image.decode_jpeg(content, channels),
                lambda: tf.image.decode_png(content, channels))

        features = {
            'image/encoded':
                tf.io.FixedLenFeature((), tf.string, default_value=''),
            'image/filename':
                tf.io.FixedLenFeature((), tf.string, default_value=''),
            'image/format':
                tf.io.FixedLenFeature((), tf.string, default_value='jpeg'),
            'image/height':
                tf.io.FixedLenFeature((), tf.int64, default_value=0),
            'image/width':
                tf.io.FixedLenFeature((), tf.int64, default_value=0),
            'image/segmentation/class/encoded':
                tf.io.FixedLenFeature((), tf.string, default_value=''),
            'image/segmentation/class/format':
                tf.io.FixedLenFeature((), tf.string, default_value='png'),
        }

        parsed_features = tf.io.parse_single_example(example_proto, features)

        image = _decode_image(parsed_features['image/encoded'], channels=3)

        label = _decode_image(
            parsed_features['image/segmentation/class/encoded'], channels=1)

        image_name = parsed_features['image/filename']
        if image_name is None:
            image_name = tf.constant('')

        sample = {
            common.IMAGE: image,
            common.IMAGE_NAME: image_name,
            common.HEIGHT: parsed_features['image/height'],
            common.WIDTH: parsed_features['image/width'],
        }

        if label is not None:
            if label.get_shape().ndims == 2:
                label = tf.expand_dims(label, 2)
            elif label.get_shape().ndims == 3 and label.shape.dims[2] == 1:
                pass
            else:
                raise ValueError('Input label shape must be [height, width], or '
                                 '[height, width, 1].')

            label.set_shape([None, None, 1])

            sample[common.LABELS_CLASS] = label

        return sample

    def _preprocess_image(self, sample):
        """Preprocesses the image and label.

        Args:
          sample: A sample containing image and label.

        Returns:
          sample: Sample with preprocessed image and label.

        Raises:
          ValueError: Ground truth label not provided during training.
        """
        image = sample[common.IMAGE]
        label = sample[common.LABELS_CLASS]

        original_image, image, label = input_preprocess.preprocess_image_and_label(
            image=image,
            label=label,
            crop_height=self.crop_size[0],
            crop_width=self.crop_size[1],
            min_resize_value=self.min_resize_value,
            max_resize_value=self.max_resize_value,
            resize_factor=self.resize_factor,
            min_scale_factor=self.min_scale_factor,
            max_scale_factor=self.max_scale_factor,
            scale_factor_step_size=self.scale_factor_step_size,
            ignore_label=self.ignore_label,
            is_training=self.is_training)

        # perform augmentation of input image
        if self.augmentation_policy and self.is_training:
            print("Use augmentation with following policy:")
            pprint.pprint(self.augmentation_policy)
            image = input_preprocess.apply_augmentation_policy(
                image=image,
                policy=self.augmentation_policy)

        image.set_shape([self.crop_size[0], self.crop_size[1], 3])

        sample[common.IMAGE] = image

        if label is not None:
            sample[common.LABEL] = label

        # Remove common.LABEL_CLASS key in the sample since it is only used to
        # derive label and not used in training and evaluation.
        sample.pop(common.LABELS_CLASS, None)

        return sample

    def get_tf_dataset(self):
        """Gets the dataset as tf.dataset.
        :return:
        """
        files = self._get_all_files()

        dataset = (
            tf.data.TFRecordDataset(files, num_parallel_reads=self.num_parallel_reads)
            .take(self.num_samples)
            .map(self._parse_function, num_parallel_calls=self.num_parallel_calls)
            .map(self._preprocess_image, num_parallel_calls=self.num_parallel_calls))

        if self.should_shuffle:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size)

        if self.should_repeat:
            dataset = dataset.repeat()  # Repeat forever for training.
        else:
            dataset = dataset.repeat(1)

        dataset = dataset.batch(self.batch_size).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset


    def _get_all_files(self):
        """Gets all the files to read data from.
        Returns:
          A list of input files.
        """
        glob_pattern = os.path.join(self.dataset_dir, "*" + self.split_name + "*" + ".tfrecord")
        paths_preprocessed = glob.glob(glob_pattern)
        paths_preprocessed = sorted(paths_preprocessed)

        print("Build {0} Dataset on Preprocessed Paths:".format(self.split_name))
        [print(str(i)) for i in paths_preprocessed]

        return paths_preprocessed
