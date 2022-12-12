import tensorflow as tf

from tfops_aug.augmentation_operations import AUGMENTATION_BY_NAME

def apply_augmentation_policy(image: tf.Tensor, policy: dict) -> tf.Tensor:
    """
    Applies the augmentation policy to the input image of type tf.Tensor
    :param image: Image as tf.Tensor with shape [h, w, 3] of dtype tf.uint8
    :param policy: Augmentation policy as dict
    :return: Augmented Image as tf.Tensor with dtype tf.uint8
    """
    assert image.dtype == tf.uint8, "Currently only images of type tf.uint8 are supported"

    num_policies = len(policy)

    random_policy = tf.random.uniform(
        shape=[], minval=0, maxval=num_policies, dtype=tf.int32)

    # take all policies and choose random policy based on idx
    for idx in range(num_policies):
        image = tf.cond(tf.equal(random_policy, idx),
                        lambda: apply_sub_policy(image, policy["sub_policy"+str(idx)]),
                        lambda: image)

    return image


def apply_sub_policy(image: tf.Tensor, sub_policy: dict) -> tf.Tensor:
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
        image = tf.clip_by_value(image, 0, 255)
        image = tf.cast(image, dtype=tf.uint8)

    return image


def validate_augmentation_policy(policy: dict) -> bool:
    """
    Validates if augmentation policy has a correct structure and naming.
    Returns True if everything is correct, otherwise False
    :param policy dict
    returns: Bool
    """
    num_policies = len(policy)
    for idx in range(num_policies):
        if "sub_policy" + str(idx) in policy:
            sub_policy = policy["sub_policy" + str(idx)]
            for idx_sub in range(len(sub_policy)):
                if "op"+str(idx_sub) in sub_policy:
                    operation = sub_policy["op" + str(idx_sub)]
                    if operation[0] not in AUGMENTATION_BY_NAME:
                        return False
                    if operation[1] > 1.0 or operation[1] < 0.0:
                        return False
                    if operation[2] > 10 or \
                       operation[2] < 0  or \
                       type(operation[2]) != int:
                        return False
                else:
                    return False
        else:
            return False
    return True
