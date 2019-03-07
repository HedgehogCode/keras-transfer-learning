"""ResNet models for Keras

Adapted from keras_applications/resnet_common.py

TODO document
"""

from keras import layers
import keras.backend as K


##############################################################################
#     MODELS
##############################################################################

def resnet_38(x):
    # NOTE: This is none of the original models from the paper
    def stack_fn(x):
        x = stack(x, 64, 3, stride1=1, name='conv2')
        x = stack(x, 128, 3, name='conv3')
        x = stack(x, 256, 3, name='conv4')
        x = stack(x, 512, 3, name='conv5')
        return x
    return resnet(x, stack_fn)


def resnet_50(x):
    def stack_fn(x):
        x = stack(x, 64, 3, stride1=1, name='conv2')
        x = stack(x, 128, 4, name='conv3')
        x = stack(x, 256, 6, name='conv4')
        x = stack(x, 512, 3, name='conv5')
        return x
    return resnet(stack_fn)(x)


def resnet_101(x):
    def stack_fn(x):
        x = stack(x, 64, 3, stride1=1, name='conv2')
        x = stack(x, 128, 4, name='conv3')
        x = stack(x, 256, 23, name='conv4')
        x = stack(x, 512, 3, name='conv5')
        return x
    return resnet(stack_fn)(x)


def resnet_152(x):
    def stack_fn(x):
        x = stack(x, 64, 3, stride1=1, name='conv2')
        x = stack(x, 128, 8, name='conv3')
        x = stack(x, 256, 36, name='conv4')
        x = stack(x, 512, 3, name='conv5')
        return x
    return resnet(stack_fn)(x)


def resnet(stack_fn):
    def build(x):
        bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

        # First conv layer
        with K.name_scope('conv1'):
            x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(x)
            x = layers.Conv2D(64, 7, strides=2, use_bias=True,
                            name='conv1_conv')(x)
            x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                        name='conv1_bn')(x)
            x = layers.Activation('relu', name='conv1_relu')(x)

        # First Max-Pool
        with K.name_scope('pool1'):
            x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
            x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

        # Residual stacks
        x = stack_fn(x)

    return build


##############################################################################
#     BLOCK
##############################################################################

def block(x, filters, kernel_size=3, stride=1,
          conv_shortcut=True, name=None):
    """A residual block.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.

    # Returns
        Output tensor for the residual block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

    with K.name_scope(name):
        if conv_shortcut is True:
            shortcut = layers.Conv2D(4 * filters, 1, strides=stride,
                                     name=name + '_0_conv')(x)
            shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                                 name=name + '_0_bn')(shortcut)
        else:
            shortcut = x

        x = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                      name=name + '_1_bn')(x)
        x = layers.Activation('relu', name=name + '_1_relu')(x)

        x = layers.Conv2D(filters, kernel_size, padding='SAME',
                          name=name + '_2_conv')(x)
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                      name=name + '_2_bn')(x)
        x = layers.Activation('relu', name=name + '_2_relu')(x)

        x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                      name=name + '_3_bn')(x)

        x = layers.Add(name=name + '_add')([shortcut, x])
        x = layers.Activation('relu', name=name + '_out')(x)

    return x


##############################################################################
#     STACK
##############################################################################

def stack(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.

    # Returns
        Output tensor for the stacked blocks.
    """
    with K.name_scope(name):
        x = block(x, filters, stride=stride1, name=name + '_block1')
        for i in range(2, blocks + 1):
            x = block(x, filters, conv_shortcut=False,
                      name=name + '_block' + str(i))
    return x
