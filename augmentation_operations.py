import tensorflow as tf


def apply_augmentation_policy(image, policy):
    """
    Applies the augmentation policy to the input image.
    :param image: Image as tf.tensor with dtype tf.uint8
    :param policy: Augmentation policy as JSON
    :return: Augmented Image as tf.tensor with dtype tf.uint8
    """
    number_of_policies = len(policy)

    random_policy = tf.random.uniform(
        shape=[], minval=0, maxval=number_of_policies, dtype=tf.int32)

    # take all policies and choose random policy based on idx
    for idx in range(number_of_policies):
        image = tf.cond(tf.equal(random_policy, idx),
                        lambda: apply_sub_policy(image, policy["sub_policy"+str(idx)]),
                        lambda: image)

    return image


def apply_sub_policy(image, sub_policy):
    """
    Applies a sub-policy to an input image
    :param image: Image as tf.tensor
    :param sub_policy: Sub-policy consisting of at least one operation
    :return: Augmented Image as tf.tensor (tf.int32)
    """
    for idx in range(len(sub_policy)):
        operation = sub_policy["op"+str(idx)]

        op_func = AUGMENTATION_BY_NAME[operation[0]]  # convert op string to callable function
        prob = operation[1]  # get probability
        level = operation[2]  # get level of operation

        image = tf.cond(tf.random.uniform([], 0, 1) >= (1. - prob),
                        lambda: op_func(image, level),
                        lambda: image)
        # some devices crash using tf.uint8
        image = tf.cast(image, dtype=tf.int32)
        image = tf.clip_by_value(image, 0, 255)

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
    tensor = tensor / 255
    tensor *= levels
    tensor = tf.floor(tensor)
    tensor /= levels
    tensor = tensor * 255
    """
    image = tf.cast(image, dtype=tf.float32) * (levels / 255.0)
    image = tf.round(image)
    image = image * (255.0 / levels)
    return image


def posterize(image, level):
    level = 16 - int_parameter(level, 10)
    return _posterize(image, level)


def _solarize(image, threshold):
    """
    Invert all pixel values above a threshold.
    :param image: Image as tf.tensor
    :param threshold: Threshold in [0, 255]
    :return: Solarized Image
    """
    mask = tf.constant(255, dtype=tf.int32) * tf.cast(tf.greater(image, threshold * tf.ones_like(image)), dtype=tf.int32)
    image = tf.abs(mask - tf.cast(image, dtype=tf.int32))
    return image


def solarize(image, level):
    level = 250 - int_parameter(level, 250)
    return _solarize(image, level)


def _unbiased_gamma_sampling(image, z_range):
    # Unbiased gamma sampling
    # See Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes
    # CVPR'17 for a discussion on this.
    scaled_image = tf.cast(image, dtype=tf.float32) / 255.0
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
    :return: Inverted Image as tf.uint8
    """
    image = tf.constant(255, dtype=tf.uint8) - image
    return image


def adjust_brightness(image, level):
    level = float_parameter(level, 0.9) - 0.2
    return tf.image.adjust_brightness(image, level)


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
    level = float_parameter(level, 0.8) + 0.5  # range 0.5 - 1.3
    return tf.image.adjust_gamma(image, gamma=level)


def adjust_jpeg_quality(image, level):
    level = int_parameter(level, 70)
    image = tf.cast(image, tf.float32)
    return tf.image.adjust_jpeg_quality(image / 255.0, level) * 255.0


def add_gaussian_noise(image, level):
    level = float_parameter(level, 22) + 3
    image = tf.cast(image, dtype=tf.float32)
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
    "add_noise": add_gaussian_noise
}

ALL_AUGMENTATION_NAMES = AUGMENTATION_BY_NAME.keys()

ALL_AUGMENTATION_NAMES_AS_LIST = list(AUGMENTATION_BY_NAME.keys())
