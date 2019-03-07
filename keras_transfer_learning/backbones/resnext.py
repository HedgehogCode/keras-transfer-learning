"""ResNeXt models for Keras

Adapted from keras_applications/resnet_common.py

TODO document
"""

from keras import layers
import keras.backend as K


##############################################################################
#     MODELS
##############################################################################

def resnext_50(x):
    def stack_fn(x):
        x = stack(x, 128, 3, stride1=1, name='conv2')
        x = stack(x, 256, 4, name='conv3')
        x = stack(x, 512, 6, name='conv4')
        x = stack(x, 1024, 3, name='conv5')
        return x
    return resnext(stack_fn)(x)


def resnext_101(x):
    def stack_fn(x):
        x = stack(x, 128, 3, stride1=1, name='conv2')
        x = stack(x, 256, 4, name='conv3')
        x = stack(x, 512, 23, name='conv4')
        x = stack(x, 1024, 3, name='conv5')
        return x
    return resnext(stack_fn)(x)


def resnext(x, stack_fn):
    def build(x):
        bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

        # First conv layer
        with K.name_scope('conv1'):
            x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(x)
            x = layers.Conv2D(64, 7, strides=2, use_bias=False,
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

def block(x, filters, kernel_size=3, stride=1, groups=32,
          conv_shortcut=True, name=None):
    """A residual block.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        groups: default 32, group size for grouped convolution.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.

    # Returns
        Output tensor for the residual block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

    with K.name_scope(name):
        if conv_shortcut is True:
            shortcut = layers.Conv2D((64 // groups) * filters, 1, strides=stride,
                                     use_bias=False, name=name + '_0_conv')(x)
            shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                                 name=name + '_0_bn')(shortcut)
        else:
            shortcut = x

        x = layers.Conv2D(filters, 1, use_bias=False, name=name + '_1_conv')(x)
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                      name=name + '_1_bn')(x)
        x = layers.Activation('relu', name=name + '_1_relu')(x)

        c = filters // groups
        x = layers.ZeroPadding2D(
            padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
        x = layers.DepthwiseConv2D(kernel_size, strides=stride, depth_multiplier=c,
                                   use_bias=False, name=name + '_2_conv')(x)
        x_shape = K.int_shape(x)[1:-1]
        x = layers.Reshape(x_shape + (groups, c, c))(x)
        output_shape = x_shape + \
            (groups, c) if K.backend() == 'theano' else None
        x = layers.Lambda(lambda x: sum([x[:, :, :, :, i] for i in range(c)]),
                          output_shape=output_shape, name=name + '_2_reduce')(x)
        x = layers.Reshape(x_shape + (filters,))(x)
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                      name=name + '_2_bn')(x)
        x = layers.Activation('relu', name=name + '_2_relu')(x)

        x = layers.Conv2D((64 // groups) * filters, 1,
                          use_bias=False, name=name + '_3_conv')(x)
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                      name=name + '_3_bn')(x)

        x = layers.Add(name=name + '_add')([shortcut, x])
        x = layers.Activation('relu', name=name + '_out')(x)
    return x


##############################################################################
#     STACK
##############################################################################

def stack(x, filters, blocks, stride1=2, groups=32, name=None):
    """A set of stacked residual blocks.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        groups: default 32, group size for grouped convolution.
        name: string, stack label.

    # Returns
        Output tensor for the stacked blocks.
    """
    with K.name_scope(name):
        x = block(x, filters, stride=stride1,
                  groups=groups, name=name + '_block1')
        for i in range(2, blocks + 1):
            x = block(x, filters, groups=groups, conv_shortcut=False,
                      name=name + '_block' + str(i))
    return x
