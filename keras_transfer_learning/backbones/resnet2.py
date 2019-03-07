"""ResNet2 models for Keras

Adapted from keras_applications/resnet_common.py

TODO document
"""

from keras import layers
import keras.backend as K


##############################################################################
#     MODELS
##############################################################################

def resnet2_50(x):
    def stack_fn(x):
        x = stack(x, 64, 3, name='conv2')
        x = stack(x, 128, 4, name='conv3')
        x = stack(x, 256, 6, name='conv4')
        x = stack(x, 512, 3, stride1=1, name='conv5')
        return x
    return resnet2(stack_fn)(x)


def resnet2_101(x):
    def stack_fn(x):
        x = stack(x, 64, 3, name='conv2')
        x = stack(x, 128, 4, name='conv3')
        x = stack(x, 256, 23, name='conv4')
        x = stack(x, 512, 3, stride1=1, name='conv5')
        return x
    return resnet2(stack_fn)(x)


def resnet2_152(x):
    def stack_fn(x):
        x = stack(x, 64, 3, name='conv2')
        x = stack(x, 128, 8, name='conv3')
        x = stack(x, 256, 36, name='conv4')
        x = stack(x, 512, 3, stride1=1, name='conv5')
        return x
    return resnet2(stack_fn)(x)


def resnet2(stack_fn):
    def build(x):
        bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

        # First conv layer
        with K.name_scope('conv1'):
            x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(x)
            x = layers.Conv2D(64, 7, strides=2, use_bias=True,
                            name='conv1_conv')(x)

        # First Max-Pool
        with K.name_scope('pool1'):
            x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
            x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

        # Residual stacks
        x = stack_fn(x)

        # Post activation
        with K.name_scope('post'):
            x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                        name='post_bn')(x)
            x = layers.Activation('relu', name='post_relu')(x)

    return build


##############################################################################
#     BLOCK
##############################################################################

def block(x, filters, kernel_size=3, stride=1,
          conv_shortcut=False, name=None):
    """A residual block.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default False, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.

    # Returns
        Output tensor for the residual block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

    with K.name_scope(name):
        preact = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                           name=name + '_preact_bn')(x)
        preact = layers.Activation('relu', name=name + '_preact_relu')(preact)

        if conv_shortcut is True:
            shortcut = layers.Conv2D(4 * filters, 1, strides=stride,
                                     name=name + '_0_conv')(preact)
        else:
            shortcut = layers.MaxPooling2D(
                1, strides=stride)(x) if stride > 1 else x

        x = layers.Conv2D(filters, 1, strides=1, use_bias=False,
                          name=name + '_1_conv')(preact)
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                      name=name + '_1_bn')(x)
        x = layers.Activation('relu', name=name + '_1_relu')(x)

        x = layers.ZeroPadding2D(
            padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
        x = layers.Conv2D(filters, kernel_size, strides=stride,
                          use_bias=False, name=name + '_2_conv')(x)
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                      name=name + '_2_bn')(x)
        x = layers.Activation('relu', name=name + '_2_relu')(x)

        x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
        x = layers.Add(name=name + '_out')([shortcut, x])
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
        x = block(x, filters, conv_shortcut=True,
                  name=name + '_block1')
        for i in range(2, blocks):
            x = block(x, filters, name=name + '_block' + str(i))
        x = block(x, filters, stride=stride1,
                  name=name + '_block' + str(blocks))
    return x
