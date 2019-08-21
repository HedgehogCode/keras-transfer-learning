"""ResNet models for Keras

Adapted from keras_applications/resnet_common.py

TODO document
"""

from keras import layers
import keras.backend as K


##############################################################################
#     MODELS
##############################################################################

def resnet_38(y):
    # NOTE: This is none of the original models from the paper
    def stack_fn(x):
        x = stack(64, 3, stride1=1, name='conv2')(x)
        x = stack(128, 3, name='conv3')(x)
        x = stack(256, 3, name='conv4')(x)
        x = stack(512, 3, name='conv5')(x)
        return x
    return resnet(stack_fn)(y)


def resnet_50(y):
    def stack_fn(x):
        x = stack(64, 3, stride1=1, name='conv2')(x)
        x = stack(128, 4, name='conv3')(x)
        x = stack(256, 6, name='conv4')(x)
        x = stack(512, 3, name='conv5')(x)
        return x
    return resnet(stack_fn)(y)


def resnet_101(y):
    def stack_fn(x):
        x = stack(64, 3, stride1=1, name='conv2')(x)
        x = stack(128, 4, name='conv3')(x)
        x = stack(256, 23, name='conv4')(x)
        x = stack(512, 3, name='conv5')(x)
        return x
    return resnet(stack_fn)(y)


def resnet_152(y):
    def stack_fn(x):
        x = stack(64, 3, stride1=1, name='conv2')(x)
        x = stack(128, 8, name='conv3')(x)
        x = stack(256, 36, name='conv4')(x)
        x = stack(512, 3, name='conv5')(x)
        return x
    return resnet(stack_fn)(y)


def resnet(stack_fn):
    def build(x):
        bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

        # First conv layer
        with K.name_scope('conv1'):
            x = layers.ZeroPadding2D(
                padding=((3, 3), (3, 3)), name='conv1_pad')(x)
            x = layers.Conv2D(64, 7, strides=2, use_bias=True,
                              name='conv1_conv')(x)
            x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                          name='conv1_bn')(x)
            x = layers.Activation('relu', name='conv1_relu')(x)

        # First Max-Pool
        with K.name_scope('pool1'):
            x = layers.ZeroPadding2D(
                padding=((1, 1), (1, 1)), name='pool1_pad')(x)
            x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

        # Residual stacks
        x = stack_fn(x)
        return x

    return build


##############################################################################
#     BLOCK
##############################################################################

def block(filters, kernel_size=3, stride=1, conv_shortcut=True, ndims=2, name=None,
          shortcut_con=True):
    """A residual block.

    # Arguments
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        ndims: default 2, number of dimensions of the input
        name: string, block label.

    # Returns
        A function that applies the block to a tensor and returns the output tensor.
    """
    bn_axis = ndims + 1 if K.image_data_format() == 'channels_last' else 1
    conv_fn = layers.Conv2D if ndims == 2 else layers.Conv3D

    def build(x):
        with K.name_scope(name):
            if conv_shortcut is True:
                shortcut = conv_fn(4 * filters, 1, strides=stride,
                                   name=name + '_0_conv')(x)
                shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                                     name=name + '_0_bn')(shortcut)
            else:
                shortcut = x

            x = conv_fn(filters, 1, strides=stride,
                        name=name + '_1_conv')(x)
            x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                          name=name + '_1_bn')(x)
            x = layers.Activation('relu', name=name + '_1_relu')(x)

            x = conv_fn(filters, kernel_size, padding='SAME',
                        name=name + '_2_conv')(x)
            x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                          name=name + '_2_bn')(x)
            x = layers.Activation('relu', name=name + '_2_relu')(x)

            x = conv_fn(4 * filters, 1, name=name + '_3_conv')(x)
            x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                          name=name + '_3_bn')(x)

            if shortcut_con:
                x = layers.Add(name=name + '_add')([shortcut, x])
            x = layers.Activation('relu', name=name + '_out')(x)
        return x

    return build


##############################################################################
#     STACK
##############################################################################

def stack(filters, blocks, stride1=2, ndims=2, name=None, shortcut=True):
    """A set of stacked residual blocks.

    # Arguments
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.

    # Returns
        A function that applies the stacked blocks to a tensor and returns the output tensor.
    """
    def build(x):
        with K.name_scope(name):
            x = block(filters, stride=stride1, ndims=ndims,
                      name=name + '_block1', shortcut_con=shortcut)(x)
            for i in range(2, blocks + 1):
                x = block(filters, conv_shortcut=False, ndims=ndims,
                          name=name + '_block' + str(i), shortcut_con=shortcut)(x)
        return x
    return build
