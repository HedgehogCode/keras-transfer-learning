import tensorflow as tf

from keras import layers


def segm(num_classes, out_activation='softmax', feature_layer=0, feature_kernel_size=3,
         feature_activation='relu', batch_norm=False):
    """Retuns a function that adds a top to a Keras model which outputs a segmentation. The output
    segmentation has the same spatial dimensions as the input tensor.

    To be precise the created function adds a convolutional layer with a 1x1 kernel and an output
    activation to the network. If feature_layer is set to something higher than zero an additional
    convolutional layer is added before the output layer.

    Arguments:
        num_classes {int} -- number of output classes.

    Keyword Arguments:
        out_activation {str} -- activation of the output layer. (default: {'softmax'})
        feature_layer {int} -- number of filters of an additional feature layer. If zero, no
                               feature layer will be added. (default: {0})
        feature_kernel_size {int} -- the kernel size of an additional feature layer. (default: {3})
        feature_activation {str} -- the activation function of an additional feature layer.
                                    (default: {'relu'})
        batch_norm {bool} -- if a batch normalization layer should be added after the feature layer
                             (before the activation). (default: {False})

    Returns:
        function -- a function that adds the top on the given input tensor
    """

    def build(x):
        rank = x.get_shape().ndims
        conv = layers.Conv2D if rank == 4 else layers.Conv3D
        if feature_layer > 0:
            x = conv(feature_layer, feature_kernel_size, padding='same')(x)
            if batch_norm:
                x = layers.BatchNormalization()(x)
            x = layers.Activation(feature_activation)(x)

        x = conv(num_classes, 1, padding='same')(x)
        x = layers.Activation(out_activation)(x)
        return x
    return build
