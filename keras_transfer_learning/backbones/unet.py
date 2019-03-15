"""U-Net models for Keras

TODO valid paddding
"""
import keras.backend as K
from keras import layers


def unet(filters=None, kernel_size=3, activation='relu', batch_norm=False, ndims=2):
    """Creates a U-Net similar to the U-Net in the origial paper. The only difference is that
    padding is applied for convolutions. Therefore the input and output has the same size.

    Keyword Arguments:
        filters {list of ints} -- list of the filters (Determines the depth of the U-Net)
                                  (default: [64, 128, 256, 512, 1024])
        kernel_size {int or tuple} -- size of the convolutional kernels (default: {3})
        activation {str} -- name of the activation function (default: {'relu'})
        batch_norm {boolean} -- if batch normalization should be added after each convolutional
                                layer (default: {False})
        ndims {int} -- number of dimensions (default: {2})

    Returns:
        function -- a function which applies a U-Net to an input tensor
    """

    if filters is None:
        filters = [64, 128, 256, 512, 1024]

    def build(tensor):
        # Downsample
        tensors = []
        down_filters = filters[:-1]
        for idx, filt in enumerate(down_filters):
            tensor = conv_block(ndims, filt, kernel_size=kernel_size,
                                activation=activation, batch_norm=batch_norm,
                                name='features_down' + str(idx))(tensor)
            tensors.insert(0, tensor)
            tensor = downsample_block(
                ndims, name='downsample' + str(idx))(tensor)

        # Middle
        tensor = conv_block(ndims, filters[-1], kernel_size=kernel_size,
                            activation=activation, batch_norm=batch_norm,
                            name='features_middle')(tensor)

        # Upsample
        up_filters = filters[-2::-1]
        for idx, filt in enumerate(up_filters):
            tensor = upsample_block(
                ndims, filt, name='upsample' + str(idx))(tensor)
            tensor = layers.Concatenate(axis=-1, name='concat' + str(idx))([tensors.pop(0), tensor])
            tensor = conv_block(ndims, filt, kernel_size=kernel_size,
                                activation=activation, batch_norm=batch_norm,
                                name='features_up' + str(idx))(tensor)
        return tensor
    return build


def conv_block(ndims, filters, kernel_size=3, padding='same', activation='relu', batch_norm=False,
               name=None):
    """Creates a block of two convolutional layers with actitivations after the convolutions.

    Arguments:
        ndims {int} -- number of dimensions
        filters {int} -- number of filters for both convolutional layers.

    Keyword Arguments:
        kernel_size {tuple} -- size of the convolutional kernels (default: {(3, 3)})
        padding {str} -- padding of the convolutions (default: {'same'})
        activation {str} -- name of the activation function (default: {'relu'})
        batch_norm {boolean} -- if batch normalization should be added after each convolutional
                                layer (default: {False})
        name {[type]} -- name of the block (default: {None})

    Returns:
        function -- a function which applies this block to a tensor
    """
    conv_fn = layers.Conv2D if ndims == 2 else layers.Conv3D

    def build(tensor):
        with K.name_scope(name):
            tensor = conv_fn(filters, kernel_size, padding=padding,
                             name=name + '_conv1')(tensor)
            if batch_norm:
                tensor = layers.BatchNormalization(name=name + '_bn1')(tensor)
            tensor = layers.Activation(activation,
                                       name=name + '_' + activation + '1')(tensor)
            tensor = conv_fn(filters, kernel_size, padding=padding,
                             name=name + '_conv2')(tensor)
            if batch_norm:
                tensor = layers.BatchNormalization(name=name + '_bn2')(tensor)
            tensor = layers.Activation(activation,
                                       name=name + '_' + activation + '2')(tensor)
        return tensor
    return build


def upsample_block(ndims, filters, size=2, padding='same', name=None):
    """Creates an upsample block which consists of a upsampling layer and a convolutional layer
    with the number of filters.

    Arguments:
        ndims {int} -- number of dimensions
        filters {int} -- number of filters of the convolutional layer

    Keyword Arguments:
        size {tuple} -- size of the upsampling and convolutional kernels (default: {(2, 2, 2)})
        padding {str} -- padding of the convolutional layer (default: {'same'})
        name {[type]} -- name of the block (default: {None})

    Returns:
        function -- a function which applies this block to a tensor
    """
    upsampling_fn = layers.UpSampling2D if ndims == 2 else layers.UpSampling3D
    conv_fn = layers.Conv2D if ndims == 2 else layers.Conv3D

    def build(tensor):
        with K.name_scope(name):
            tensor = upsampling_fn(size, name=name + '_upsample')(tensor)
            tensor = conv_fn(filters, size, padding=padding,
                             name=name + '_conv_up')(tensor)
        return tensor
    return build


def downsample_block(ndims, size=2, name=None):
    """Creates a downsample block which consists of a max pooling layer.

    Arguments:
        ndims {int} -- number of dimensions

    Keyword Arguments:
        size {tuple} -- size of downsampling (default: {(2, 2)})
        name {[type]} -- name of the block (default: {None})

    Returns:
        function -- a function which applies this block to a tensor
    """
    maxpool_fn = layers.MaxPool2D if ndims == 2 else layers.MaxPool3D

    def build(tensor):
        with K.name_scope(name):
            tensor = maxpool_fn(size, name=name + '_maxpool')(tensor)
        return tensor
    return build
