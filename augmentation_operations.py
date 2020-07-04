import tensorflow as tf
import tensorflow_addons as tfa

import common


def apply_augmentation_policy(image, policy) -> tf.Tensor:
    """
    Applies the augmentation policy to the input image.
    :param image: Image as tf.tensor with shape [h, w, 3]
    :param policy: Augmentation policy as dict
    :return: Augmented Image as tf.tensor with dtype tf.float32
    """
    num_policies = len(policy)

    random_policy = tf.random.uniform(
        shape=[], minval=0, maxval=num_policies, dtype=tf.int32)

    # take all policies and choose random policy based on idx
    for idx in range(num_policies):
        image = tf.cond(tf.equal(random_policy, idx),
                        lambda: apply_sub_policy(image, policy["sub_policy"+str(idx)]),
                        lambda: image)

    return image


def apply_sub_policy(image, sub_policy) -> tf.Tensor:
    """
    Applies a sub-policy to an input image
    :param image: Image as tf.tensor (tf.float32)
    :param sub_policy: Sub-policy consisting of at least one operation
    :return: Augmented Image as tf.tensor (tf.float32)
    """
    for idx in range(len(sub_policy)):
        operation = sub_policy["op"+str(idx)]

        op_func = AUGMENTATION_BY_NAME[operation[0]]  # convert op string to callable function
        prob = operation[1]  # get probability
        level = operation[2]  # get level of operation

        image = tf.cond(tf.random.uniform([], 0, 1) >= (1. - prob),
                        lambda: op_func(image, level),
                        lambda: image)
        image = tf.cast(image, dtype=tf.float32)
        image = tf.clip_by_value(image, 0.0, 255.0)

    return image


def int_parameter(level, maxval) -> int:
    """Helper function to scale `val` between 0 and maxval .

    Args:
      level: Level of the operation that will be between [0, `PARAMETER_MAX`].
      maxval: Maximum value that the operation can have. This will be scaled
        to level/PARAMETER_MAX.

    Returns:
      An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval) -> float:
    """Helper function to scale `val` between 0 and maxval .

    Args:
      level: Level of the operation that will be between [0, `PARAMETER_MAX`].
      maxval: Maximum value that the operation can have. This will be scaled
        to level/PARAMETER_MAX.

    Returns:
      A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10


# Kernel Augmentations
def _sharpen(image, level) -> tf.Tensor:
    """
    Implements Sharpening Function
    :param image:
    :param level:
    :return:
    """
    orig_image = image
    # Make image 4D for conv operation.
    image = tf.expand_dims(image, 0)

    kernel = (
        tf.constant(
            [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=tf.float32, shape=[3, 3, 1, 1]
        )
        * level
    )
    # Normalize kernel
    kernel = kernel / tf.reduce_sum(kernel)
    # Tile across channel dimension.
    kernel = tf.tile(kernel, [1, 1, 3, 1])
    strides = [1, 1, 1, 1]
    convolved = tf.nn.depthwise_conv2d(
        image, kernel, strides, padding="SAME", dilations=[1, 1]
    )
    convolved = tf.clip_by_value(convolved, 0.0, 255.0)
    convolved = tf.squeeze(convolved, [0])

    blended = tfa.image.blend(convolved, orig_image, level)
    return blended


def sharpen(image, level) -> tf.Tensor:
    level = float_parameter(level, 4.5) + 1.5
    return _sharpen(image, level)


def _gaussian_blur(image, level, sigma=3) -> tf.Tensor:
    """
    Implements Gaussian Blur Function
    :param image:
    :param level:
    :return:
    """
    # Make image 4D for conv operation.
    image = tf.expand_dims(image, 0)

    kernel_size = level
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=tf.float32)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype=tf.float32), 4))))
    g_norm2d = tf.pow(tf.reduce_sum(g), 2)
    kernel = tf.tensordot(g, g, axes=0) / g_norm2d
    kernel = tf.reshape(kernel, [kernel_size, kernel_size, 1, 1])

    # Tile across channel dimension.
    kernel = tf.tile(kernel, [1, 1, 3, 1])
    strides = [1, 1, 1, 1]
    convolved = tf.nn.depthwise_conv2d(
        image, kernel, strides, padding="SAME", dilations=[1, 1]
    )
    convolved = tf.clip_by_value(convolved, 0.0, 255.0)
    convolved = tf.squeeze(convolved, [0])

    return convolved


def gaussian_blur(image, level) -> tf.Tensor:
    level = int_parameter(level, 12) + 3
    return _gaussian_blur(image, level)


# Color Augmentations
def _posterize(image, levels) -> tf.Tensor:
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


def posterize(image, level) -> tf.Tensor:
    level = 16 - int_parameter(level, 10)
    return _posterize(image, level)


def _solarize(image, threshold) -> tf.Tensor:
    """
    Invert all pixel values above a threshold.
    :param image: Image as tf.tensor
    :param threshold: Threshold in [0, 255]
    :return: Solarized Image
    """
    mask = tf.constant(255, dtype=tf.uint8) * tf.cast(tf.greater(image, threshold * tf.ones_like(image)), dtype=tf.uint8)
    image = tf.abs(tf.cast(mask, dtype=tf.float32) - image)
    return image


def solarize(image, level) -> tf.Tensor:
    level = 250 - int_parameter(level, 250)
    return _solarize(image, level)


def _unbiased_gamma_sampling(image, z_range) -> tf.Tensor:
    # Unbiased gamma sampling
    # See Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes
    # CVPR'17 for a discussion on this.
    scaled_image = tf.cast(image, dtype=tf.float32) / 255.0
    factor = tf.random.uniform(shape=[], minval=-z_range, maxval=z_range, dtype=tf.float32)
    gamma = (tf.math.log(0.5 + 1.0 / tf.sqrt(2.0) * factor) /
             tf.math.log(0.5 - 1.0 / tf.sqrt(2.0) * factor))
    image = tf.math.pow(scaled_image, gamma) * 255.0
    return image


def unbiased_gamma_sampling(image, level) -> tf.Tensor:
    level = float_parameter(level, 0.5)
    return _unbiased_gamma_sampling(image, z_range=level)


def _equalize_histogram(image) -> tf.Tensor:
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


def equalize_histogram(image, _) -> tf.Tensor:
    """
    Equalize the RGB channels of the input image
    :param image:
    :param _:
    :return:
    """
    r = _equalize_histogram(tf.expand_dims(image[:, :, 0], -1))
    g = _equalize_histogram(tf.expand_dims(image[:, :, 1], -1))
    b = _equalize_histogram(tf.expand_dims(image[:, :, 2], -1))
    image = tf.squeeze(tf.cast(tf.stack([r, g, b], axis=-2), dtype=tf.float32))
    return image


def invert(image, _) -> tf.Tensor:
    """
    Invert all pixel of the input image
    :param image: Image as tf.tensor
    :param _: Level Not used
    :return: Inverted Image as tf.float32
    """
    image = tf.constant(255.0, dtype=tf.float32) - image
    return image


def adjust_brightness(image, level) -> tf.Tensor:
    level = int_parameter(level, 180)
    return tf.image.adjust_brightness(image, level)


def adjust_contrast(image, level) -> tf.Tensor:
    level = float_parameter(level, 2) + 0.3  # with zero, image is not visible
    return tf.image.adjust_contrast(image, level)


def adjust_hue(image, level) -> tf.Tensor:
    level = float_parameter(level, 0.9)
    return tf.image.adjust_hue(image, delta=level)


def adjust_saturation(image, level) -> tf.Tensor:
    level = float_parameter(level, 2)
    return tf.image.adjust_saturation(image, saturation_factor=level)


def adjust_gamma(image, level) -> tf.Tensor:
    level = float_parameter(level, 0.8) + 0.5  # range 0.5 - 1.3
    return tf.image.adjust_gamma(image, gamma=level)


def adjust_jpeg_quality(image, level) -> tf.Tensor:
    level = int_parameter(level, 70)
    image = tf.cast(image, tf.float32)
    return tf.image.adjust_jpeg_quality(image / 255.0, level) * 255.0


def add_gaussian_noise(image, level) -> tf.Tensor:
    level = float_parameter(level, 22) + 3
    noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=level, dtype=tf.float32)
    return image + noise


# Shape Augmentations
def _shear_x(image, level) -> tf.Tensor:
    image = tfa.image.shear_x(
        tf.cast(image, dtype=tf.uint8), level, common.PIXEL_VALUE_PAD)
    return image

def shear_x(image, level) -> tf.Tensor:
    level = float_parameter(level, 0.75) - 0.75/2
    image = _shear_x(image, level)
    return image

def _shear_y(image, level) -> tf.Tensor:
    image = tfa.image.shear_y(
        tf.cast(image, dtype=tf.uint8), level, common.PIXEL_VALUE_PAD)
    return image

def shear_y(image, level) -> tf.Tensor:
    level = float_parameter(level, 0.75) - 0.75/2
    image = _shear_y(image, level)
    return image

def _translate_x(image, level) -> tf.Tensor:
    image = tfa.image.translate_xy(
        tf.cast(image, dtype=tf.uint8),
        [level, 0],
        common.PIXEL_VALUE_PAD
    )
    return image

def translate_x(image, level) -> tf.Tensor:
    level = int_parameter(level, 300) - 150
    image = _translate_x(image, level)
    return image

def _translate_y(image, level) -> tf.Tensor:
    image = tfa.image.translate_xy(
        tf.cast(image, dtype=tf.uint8),
        [0, level],
        common.PIXEL_VALUE_PAD
    )
    return image

def translate_y(image, level) -> tf.Tensor:
    level = int_parameter(level, 300) - 150
    image = _translate_y(image, level)
    return image


AUGMENTATION_BY_NAME = {
    "sharpen": sharpen,
    "gaussian_blur": gaussian_blur,
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
    "add_noise": add_gaussian_noise,
    "shear_x": shear_x,
    "shear_y": shear_y,
    "translate_x": translate_x,
    "translate_y": translate_y
}

ALL_AUGMENTATION_NAMES = AUGMENTATION_BY_NAME.keys()

ALL_AUGMENTATION_NAMES_AS_LIST = list(AUGMENTATION_BY_NAME.keys())
