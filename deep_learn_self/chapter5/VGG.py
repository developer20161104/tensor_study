#
# vgg contains consecutive conv and pooling layer
# with same structure
#

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16


def vgg_block(num_convs, num_channels):
    blk = tf.keras.models.Sequential()

    for _ in range(num_convs):
        blk.add(
            tf.keras.layers.Conv2D(
                num_channels,
                kernel_size=3,
                padding='same',
                activation='relu',
            )
        )

        blk.add(
            tf.keras.layers.MaxPool2D(
                pool_size=2,
                strides=2
            )
        )

    return blk


def vgg(conv_arch):
    net = tf.keras.models.Sequential()

    # net.add(
    #     tf.keras.layers.Input(shape=(224, 224, 1))
    # )

    # repeat same blocks
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))

    net.add(tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(4096, activation='relu'),
                                        tf.keras.layers.Dropout(0.5),
                                        tf.keras.layers.Dense(4096, activation='relu'),
                                        tf.keras.layers.Dropout(0.5),
                                        tf.keras.layers.Dense(10, activation='sigmoid')]))

    return net


conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
model = vgg(conv_arch)

X = tf.random.uniform((1, 224, 224, 1))
for blk in model.layers:
    X = blk(X)
    print(blk.name, 'output shape:\t', X.shape)

vgg_model = VGG16(weights='imagenet')
vgg_model.summary()