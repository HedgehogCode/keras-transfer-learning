"""U-Net models for Keras

TODO document
TODO add name scopes everywhere
TODO valid paddding
"""
import keras.backend as K
from keras import layers


def unet():
    # TODO implement unet from csbdeep
    filters = [64, 128, 256, 512, 1024]
    kernel_size = (3, 3)
    activation = 'relu'

    def build(x):
        # Downsample
        xs = []
        down_filters = filters[:-1]
        for idx, f in enumerate(down_filters):
            x = conv_block_2d(f, kernel_size=kernel_size,
                              activation=activation, name='conv_down_' + str(idx))(x)
            x = downsample_block_2d(name='downsample_' + str(idx))(x)
            xs.insert(0, x)

        # Middle
        x = conv_block_2d(f, kernel_size=kernel_size,
                          activation=activation, name='conv_middle_' + str(idx))(x)

        # Upsample
        up_filters = filters[-2::-1]
        for idx, f in enumerate(up_filters):
            x = upsample_block_2d(f, name='upsample_' + str(idx))(x)
            x = layers.Concatenate()([xs.pop(0), x])
            x = conv_block_2d(f, kernel_size=kernel_size,
                              activation=activation, name='conv_up_' + str(idx))(x)
        return x
    return build


def conv_block_2d(filters, kernel_size=(3, 3), padding='same', activation='relu', name=None):
    # TODO name and scope
    def build(x):
        with K.name_scope(name):
            x = layers.Conv2D(64, (3, 3), padding=padding)(x)
            x = layers.Activation(activation)(x)
            x = layers.Conv2D(64, (3, 3), padding=padding)(x)
            x = layers.Activation(activation)(x)
        return x
    return build


def upsample_block_2d(filters, size=(2, 2), padding='same', name=None):
    # TODO name and scope
    def build(x):
        with K.name_scope(name):
            x = layers.UpSampling2D((2, 2))(x)
            x = layers.Conv2D(512, (2, 2), padding='same')(x)
        return x
    return build


def downsample_block_2d(size=(2, 2), name=None):
    def build(x):
        with K.name_scope(name):
            x = layers.MaxPool2D((2, 2))(x)
        return x
    return build
