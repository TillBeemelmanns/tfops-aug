import tensorflow as tf
import tensorflow_addons as tfa

import common


def int_parameter(level: int, maxval: int, minval: int) -> int:
    """Helper function to scale `val` between minval and maxval .

    Args:
        level: Level of the operation that will be between [0, `PARAMETER_MAX`].
        maxval: Maximum value that the operation can have. This will be scaled
            to level/PARAMETER_MAX.
        minval: Minimum value that the operation can have.

    Returns:
      An int that results from scaling between maxval and minval according to `level`.
    """
    return int(maxval * (level / 10) +  ((10 - level) / 10) * minval)


def float_parameter(level: int, maxval: float, minval: float) -> float:
    """Helper function to scale `val` between 0 and maxval .

    Args:
        level: Level of the operation that will be between [0, `PARAMETER_MAX`].
        maxval: Maximum value that the operation can have. This will be scaled
        to level/PARAMETER_MAX.
        minval: Minimum value that the operation can have.

    Returns:
      A float that results from scaling between maxval and minval according to `level`.
    """
    return maxval * (level / 10) +  ((10 - level) / 10) * minval


# Kernel Augmentations
def _sharpen(image: tf.Tensor, level: float) -> tf.Tensor:
    """
    Implements sharpening function
    :param image: image of type tf.Tensor with dtype tf.float32 in range (0, 255)
    :param level: Intensity of the sharpen function
    :return: tf.Tensor with same shape as image, sharpened image
    """
    orig_image = image
    # Make image 4D for conv operation.
    image = tf.expand_dims(image, 0)

    kernel = (
        tf.constant(
            [[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=tf.float32, shape=[3, 3, 1, 1]
        )
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


def sharpen(image: tf.Tensor, level: int) -> tf.Tensor:
    level = float_parameter(level, maxval=8, minval=1.5)
    return _sharpen(image, level)


def _gaussian_blur(image: tf.Tensor, level: int, sigma=3) -> tf.Tensor:
    """
    Implements Gaussian Blur Function
    :param image: image of type tf.Tensor with dtype tf.float32 in range (0, 255)
    :param level: Intensity of the blur function
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


def gaussian_blur(image: tf.Tensor, level: int) -> tf.Tensor:
    level = int_parameter(level, maxval=15, minval=3)
    return _gaussian_blur(image, level)


# Color Augmentations
def _posterize(image: tf.Tensor, level: int) -> tf.Tensor:
    """
    Reduce the number of color levels per channel.
    :param Tensor image: image as Tensor of type tf.uint8 with range 0-255
    :param int level: level of posterize effect
    :return: Tensor
    """
    image = tf.cast(image, dtype=tf.float32) * (level / 255.0)
    image = tf.round(image)
    image = image * (255.0 / level)
    return image


def posterize(image: tf.Tensor, level: int,
              maxval=20, minval=6) -> tf.Tensor:
    level = int_parameter(level, maxval, minval)
    return _posterize(image, level)


def _solarize(image: tf.Tensor, threshold: int) -> tf.Tensor:
    """
    Invert all pixel values above a threshold
    :param image: Image as tf.Tensor
    :param threshold: Threshold in [0, 255]
    :return: Solarized Image
    """
    return tf.where(image < threshold, image, 255 - image)


def solarize(image: tf.Tensor, level: int) -> tf.Tensor:
    level = int_parameter(level, maxval=220, minval=10)
    return _solarize(image, level)


def _unbiased_gamma_sampling(image: tf.Tensor, z_range: float) -> tf.Tensor:
    # Unbiased gamma sampling
    # See Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes
    # CVPR'17 for a discussion on this.
    scaled_image = tf.cast(image, dtype=tf.float32) / 255.0
    factor = tf.random.uniform(shape=[], minval=-z_range, maxval=z_range, dtype=tf.float32)
    gamma = (tf.math.log(0.5 + 1.0 / tf.sqrt(2.0) * factor) /
             tf.math.log(0.5 - 1.0 / tf.sqrt(2.0) * factor))
    image = tf.math.pow(scaled_image, gamma) * 255.0
    return image


def unbiased_gamma_sampling(image: tf.Tensor, level: int) -> tf.Tensor:
    level = float_parameter(level, maxval=0.5, minval=0)
    return _unbiased_gamma_sampling(image, z_range=level)


def _equalize_histogram(image: tf.Tensor) -> tf.Tensor:
    """
    Normalizes the histrogram for each color channel. Assumes RGB image
    Based on the implementation presented on
    https://stackoverflow.com/questions/42835247/how-to-implement-histogram-equalization-for-images-in-tensorflow?rq=1

    :param image: Image as tf.Tensor
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


def equalize_histogram(image: tf.Tensor, level: int) -> tf.Tensor:
    r = _equalize_histogram(tf.expand_dims(image[:, :, 0], -1))
    g = _equalize_histogram(tf.expand_dims(image[:, :, 1], -1))
    b = _equalize_histogram(tf.expand_dims(image[:, :, 2], -1))
    image = tf.squeeze(tf.cast(tf.stack([r, g, b], axis=-2), dtype=tf.float32))
    return image


def invert(image: tf.Tensor, level: int) -> tf.Tensor:
    """
    Invert all pixel of the input image
    :param image: Image as tf.Tensor as tf.float32 in range (0- 255)
    :param _: Level Not used
    :return: Inverted Image as tf.float32
    """
    image = tf.constant(255.0, dtype=tf.float32) - image
    return image


def adjust_brightness(image: tf.Tensor, level: int) -> tf.Tensor:
    level = int_parameter(level, maxval=180, minval=0)
    return tf.image.adjust_brightness(image, level)


def adjust_contrast(image: tf.Tensor, level: int) -> tf.Tensor:
    level = float_parameter(level, maxval=2.3, minval=0.3)
    return tf.image.adjust_contrast(image, level)


def adjust_hue(image: tf.Tensor, level: int) -> tf.Tensor:
    level = float_parameter(level, maxval=0.9, minval=0)
    return tf.image.adjust_hue(image, delta=level)


def adjust_saturation(image: tf.Tensor, level: int) -> tf.Tensor:
    level = float_parameter(level, maxval=2, minval=0)
    return tf.image.adjust_saturation(image, saturation_factor=level)


def adjust_gamma(image: tf.Tensor, level: int) -> tf.Tensor:
    level = float_parameter(level, maxval=1.4, minval=0.7)
    return tf.image.adjust_gamma(image, gamma=level)


def adjust_jpeg_quality(image: tf.Tensor, level: int) -> tf.Tensor:
    level = int_parameter(level, maxval=50, minval=0)
    return tf.image.adjust_jpeg_quality(tf.cast(image, tf.float32) / 255.0, level) * 255.0


def add_noise(image: tf.Tensor, level: int) -> tf.Tensor:
    level = float_parameter(level, maxval=25, minval=3)
    noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=level, dtype=tf.float32)
    return image + noise


# Shape Augmentations
def _shear_x(image: tf.Tensor, level: float) -> tf.Tensor:
    image = tfa.image.shear_x(
        image, level, common.PIXEL_VALUE_PAD)
    return image


def shear_x(image: tf.Tensor, level: int) -> tf.Tensor:
    level = float_parameter(level, maxval=0.375, minval=-0.375)
    image = _shear_x(image, level)
    return image


def _shear_y(image: tf.Tensor, level: float) -> tf.Tensor:
    image = tfa.image.shear_y(
        image, level, common.PIXEL_VALUE_PAD)
    return image


def shear_y(image: tf.Tensor, level: int) -> tf.Tensor:
    level = float_parameter(level, minval=0.375, maxval=-0.375)
    image = _shear_y(image, level)
    return image


def _translate_x(image: tf.Tensor, level: float) -> tf.Tensor:
    image = tfa.image.translate_xy(
        image,
        [level, 0],
        common.PIXEL_VALUE_PAD
    )
    return image


def translate_x(image: tf.Tensor, level: int) -> tf.Tensor:
    level = int_parameter(level, maxval=150, minval=-150)
    image = _translate_x(image, level)
    return image


def _translate_y(image: tf.Tensor, level: int) -> tf.Tensor:
    image = tfa.image.translate_xy(
        image,
        [0, level],
        common.PIXEL_VALUE_PAD
    )
    return image


def translate_y(image: tf.Tensor, level: int) -> tf.Tensor:
    level = int_parameter(level, maxval=150, minval=-150)
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
    "add_noise": add_noise,
    "shear_x": shear_x,
    "shear_y": shear_y,
    "translate_x": translate_x,
    "translate_y": translate_y
}

ALL_AUGMENTATION_NAMES = AUGMENTATION_BY_NAME.keys()

ALL_AUGMENTATION_NAMES_AS_LIST = list(AUGMENTATION_BY_NAME.keys())
