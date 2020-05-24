import tensorflow as tf
import preprocess_utils

# Constants
_PROB_OF_FLIP = 0.5
_MEAN_PIXEL_VALUE_PAD = [127.5, 127.5, 127.5]

def preprocess_image_and_label(image,
                               label,
                               crop_height,
                               crop_width,
                               min_resize_value=None,
                               max_resize_value=None,
                               resize_factor=None,
                               min_scale_factor=1.,
                               max_scale_factor=1.,
                               scale_factor_step_size=0,
                               ignore_label=255,
                               is_training=True):
    """Preprocesses the image and label.
    Args:
        image: Input image.
        label: Ground truth annotation label.
        crop_height: The height value used to crop the image and label.
        crop_width: The width value used to crop the image and label.
        min_resize_value: Desired size of the smaller image side.
        max_resize_value: Maximum allowed size of the larger image side.
        resize_factor: Resized dimensions are multiple of factor plus one.
        min_scale_factor: Minimum scale factor value.
        max_scale_factor: Maximum scale factor value.
        scale_factor_step_size: The step size from min scale factor to max scale
          factor. The input is randomly scaled based on the value of
          (min_scale_factor, max_scale_factor, scale_factor_step_size).
        ignore_label: The label value which will be ignored for training and
          evaluation.
        is_training: If the preprocessing is used for training or not.
        model_variant: Model variant (string) for choosing how to mean-subtract the
          images. See feature_extractor.network_map for supported model variants.

    Returns:
        original_image: Original image (could be resized).
        processed_image: Preprocessed image.
        label: Preprocessed ground truth segmentation label.

    Raises:
        ValueError: Ground truth label not provided during training.
    """
    if is_training and label is None:
        raise ValueError('Label not provided, but necessary during training.')

    # Keep reference to original image.
    original_image = image

    processed_image = tf.cast(image, tf.float32)

    if label is not None:
        label = tf.cast(label, tf.int32)

    # Resize image and label to the desired range.
    if min_resize_value or max_resize_value:
        [processed_image, label] = (
            preprocess_utils.resize_to_range(
                image=processed_image,
                label=label,
                min_size=min_resize_value,
                max_size=max_resize_value,
                factor=resize_factor,
                align_corners=True))
        # The `original_image` becomes the resized image.
        original_image = tf.identity(processed_image)

    # Data augmentation by randomly scaling the inputs.
    if is_training:
        scale = preprocess_utils.get_random_scale(
            min_scale_factor, max_scale_factor, scale_factor_step_size)
        processed_image, label = preprocess_utils.randomly_scale_image_and_label(
            processed_image, label, scale)
        processed_image.set_shape([None, None, 3])

    # Pad image and label to have dimensions >= [crop_height, crop_width]
    image_shape = tf.shape(processed_image)
    image_height = image_shape[0]
    image_width = image_shape[1]

    target_height = image_height + tf.maximum(crop_height - image_height, 0)
    target_width = image_width + tf.maximum(crop_width - image_width, 0)

    # Pad image with mean pixel value.
    mean_pixel = tf.reshape(_MEAN_PIXEL_VALUE_PAD, [1, 1, 3])
    processed_image = preprocess_utils.pad_to_bounding_box(
        processed_image, 0, 0, target_height, target_width, mean_pixel)

    if label is not None:
        label = preprocess_utils.pad_to_bounding_box(
            label, 0, 0, target_height, target_width, ignore_label)

    # Randomly crop the image and label.
    if is_training and label is not None:
        processed_image, label = preprocess_utils.random_crop(
            [processed_image, label], crop_height, crop_width)
        processed_image.set_shape([crop_height, crop_width, 3])

    if label is not None:
        label.set_shape([crop_height, crop_width, 1])

    if is_training:
        # Randomly left-right flip the image and label.
        processed_image, label, _ = preprocess_utils.flip_dim(
            [processed_image, label], _PROB_OF_FLIP, dim=1)

    return original_image, processed_image, label


def apply_augmentation_policy(image, policy):
    """
    Applies the augmentation policy to the input image.
    :param image: Image as tf.tensor
    :param policy: Augmentation policy as JSON
    :return: Augmented Image
    """
    number_of_policies = len(policy)

    random_policy = tf.random.uniform(
        shape=[], minval=0, maxval=number_of_policies, dtype=tf.int32)

    # take all policies and choose random policy based on idx
    for idx in range(number_of_policies):
        image = tf.cond(tf.equal(random_policy, idx),
                        lambda: apply_sub_policy(image, policy["sub_policy"+str(idx)]),
                        lambda: image)
    # clip values in image
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image


def apply_sub_policy(image, sub_policy):
    """
    Applies a sub-policy to an input image
    :param image: Image as tf.tensor
    :param sub_policy: Sub-policy consisting of two operations
    :return: Augmented Image
    """

    for idx in range(len(sub_policy)):
        operation = sub_policy["op"+str(idx+1)]

        op_func = AUGMENTATION_BY_NAME[operation[0]]  # convert op string to  callable function
        prob = operation[1]  # get probability
        level = operation[2]  # get level of operation

        image = tf.cond(tf.random.uniform([], 0, 1) >= (1. - prob),
                        lambda: op_func(image, level),
                        lambda: image)
        image = tf.clip_by_value(image, 0.0, 255.0)

    return image


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .

    Args:
      level: Level of the operation that will be between [0, `PARAMETER_MAX`].
      maxval: Maximum value that the operation can have. This will be scaled
        to level/PARAMETER_MAX.

    Returns:
      An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .

    Args:
      level: Level of the operation that will be between [0, `PARAMETER_MAX`].
      maxval: Maximum value that the operation can have. This will be scaled
        to level/PARAMETER_MAX.

    Returns:
      A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10


# Color Augmentations
def _posterize(image, levels):
    """
    Reduce the number of color levels per channel.

    :param Tensor image:
    :param int levels:
    :return: Tensor

    Slow, but understandable procedure
    tensor = tensor / 255.0
    tensor *= levels
    tensor = tf.floor(tensor)
    tensor /= levels
    tensor = tensor * 255.0
    """
    image = image * (levels / 255.0)
    image = tf.round(image)
    image = image * (255.0 / levels)
    return image


def posterize(image, level):
    level = int_parameter(level, 10)
    return _posterize(image, 16 - level)


def _solarize(image, threshold):
    """
    Invert all pixel values above a threshold.
    :param image: Image as tf.tensor
    :param threshold: Threshold in [0,255]
    :return: Solarized Image
    """
    mask = tf.greater(image, threshold * tf.ones_like(image))
    image = tf.abs(255.0 * tf.cast(mask, tf.float32) - image)
    return image


def solarize(image, level):
    level = int_parameter(level, 256)
    return _solarize(image, 256 - level)


def _unbiased_gamma_sampling(image, z_range):
    # Unbiased gamma sampling
    # See Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes
    # CVPR'17 for a discussion on this.
    scaled_image = image / 255.0
    factor = tf.random.uniform(shape=[], minval=-z_range, maxval=z_range, dtype=tf.float32)
    gamma = (tf.math.log(0.5 + 1.0 / tf.sqrt(2.0) * factor) /
             tf.math.log(0.5 - 1.0 / tf.sqrt(2.0) * factor))
    image = tf.math.pow(scaled_image, gamma) * 255.0
    return image


def unbiased_gamma_sampling(image, level):
    level = float_parameter(level, 0.5)
    return _unbiased_gamma_sampling(image, z_range=level)


def _equalize_histogram(image):
    """
    Based on the implementation in
    https://stackoverflow.com/questions/42835247/how-to-implement-histogram-equalization-for-images-in-tensorflow?rq=1
    """
    values_range = tf.constant([0., 255.], dtype=tf.float32)
    histogram = tf.histogram_fixed_width(tf.cast(image, tf.float32), values_range, 256)
    cdf = tf.cumsum(histogram)
    cdf_min = cdf[tf.reduce_min(tf.where(tf.greater(cdf, 0)))]

    img_shape = tf.shape(image)
    pix_cnt = img_shape[-3] * img_shape[-2]
    px_map = tf.round(tf.cast(cdf - cdf_min, dtype=tf.float32) * 255. / tf.cast(pix_cnt - 1, dtype=tf.float32))
    px_map = tf.cast(px_map, tf.uint8)

    eq_hist = tf.expand_dims(tf.gather_nd(px_map, tf.cast(image, tf.int32)), 2)
    return eq_hist


def equalize_histogram(image, _):
    # perform clipping to prevent _equalize_histogram from crashing
    image = tf.clip_by_value(image, 0, 255)

    r = _equalize_histogram(tf.expand_dims(image[:, :, 0], -1))
    g = _equalize_histogram(tf.expand_dims(image[:, :, 1], -1))
    b = _equalize_histogram(tf.expand_dims(image[:, :, 2], -1))
    image = tf.squeeze(tf.cast(tf.stack([r, g, b], axis=-2), dtype=tf.float32))
    return image


def invert(image, _):
    """
    Invert all pixel of the input image
    :param image: Image as tf.tensor
    :param _: Level Not used
    :return: Inverted Image
    """
    image = 255.0 - image
    return image


def adjust_brightness(image, level):
    level = float_parameter(level, 100)
    return tf.image.adjust_brightness(image, level - 50)


def adjust_contrast(image, level):
    level = float_parameter(level, 2) + 0.3  # with zero, image is not visible
    return tf.image.adjust_contrast(image, level)


def adjust_hue(image, level):
    level = float_parameter(level, 0.9)
    return tf.image.adjust_hue(image, delta=level)


def adjust_saturation(image, level):
    level = float_parameter(level, 2)
    return tf.image.adjust_saturation(image, saturation_factor=level)


def adjust_gamma(image, level):
    level = float_parameter(level, 0.4) + 0.8  # range 0.8 - 1.2
    return tf.image.adjust_gamma(image, gamma=level)


def adjust_jpeg_quality(image, level):
    level = int_parameter(level, 70)
    return tf.image.adjust_jpeg_quality(image / 255.0, level) * 255.0


def add_noise(image, level):
    level = float_parameter(level, 25)
    noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=level, dtype=tf.float32)
    return image + noise


AUGMENTATION_BY_NAME = {
    "posterize": posterize,
    "solarize": solarize,
    "invert": invert,
    "equalize": equalize_histogram,
    "unbiased_gamma_sampling": unbiased_gamma_sampling,
    "adjust_brightness": adjust_brightness,
    "adjust_contrast": adjust_contrast,
    "adjust_hue": adjust_hue,
    "adjust_saturation": adjust_saturation,
    "adjust_gamma": adjust_gamma,
    "adjust_jpeg_quality": adjust_jpeg_quality,
    "add_noise": add_noise
}

ALL_AUGMENTATION_NAMES = AUGMENTATION_BY_NAME.keys()

ALL_AUGMENTATION_NAMES_AS_LIST = list(AUGMENTATION_BY_NAME.keys())
