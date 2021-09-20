import tensorflow as tf
from tensorflow.keras import layers

from augmentation_policies import classification_policy
from augmentation_operations import apply_augmentation_policy


def create_classifier(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.experimental.preprocessing.Rescaling(1. / 255)(inputs)

    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return tf.keras.Model(inputs, outputs)


def augmentor_func(img, label):
    img = apply_augmentation_policy(img, classification_policy)
    return img, label


def train_classifier():
    image_size = (180, 180)
    batch_size = 64
    epochs = 50

    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "src/PetImages",
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=image_size,
        batch_size=1
    ).unbatch()
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "src/PetImages",
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )

    train_dataset = train_dataset.map(augmentor_func).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    model = create_classifier(input_shape=image_size + (3,), num_classes=2)

    callbacks = [
        tf.keras.callbacks.TensorBoard(
                        log_dir='/src/logs',
                        write_graph=True,
                        write_images=False,
                        write_steps_per_second=False,
                        update_freq='epoch',
                        profile_batch=500
        )
    ]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_dataset,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_dataset,
    )


if __name__ == '__main__':
    train_classifier()
