"""Simple convolutional neural network backbone.
"""
import keras.backend as K
from keras import layers


def convnet(filters: list = None, conv_per_block: int = 2, kernel_size=3, activation='relu',
            batch_norm: bool = False, end_maxpool: bool = False, end_global_maxpool: bool = True,
            ndims: int = 2):
    """Creates a simple convolutional neural network. The architecture is blocks of convolutional
    layers followed by max pooling layers.

    Keyword Arguments:
        filters {list of ints} -- list of the filters (Determines the depth of the network)
                                  (default: [32, 64])
        conv_per_block {int} -- number of convolutional layers per block. (default: 2)
        kernel_size {int or tuple} -- size of the convolutional kernels (default: {3})
        activation {str} -- name of the activation function (default: {'relu'})
        batch_norm {boolean} -- if batch normalization should be added after each convolutional
                                layer (default: {False})
        end_maxpool {boolean} -- if a max pooling layer should be added to the end of the network
                                 (default: {False})
        end_global_maxpool {boolean} -- if a global max pooling layer should be added to the end of
                                        the network (default: {True})
        ndims {int} -- number of dimensions (default: {2})

    Returns:
        function -- a function which applies a U-Net to an input tensor
    """

    # Check arguments
    if filters is None:
        filters = [32, 64]
    if end_maxpool and end_global_maxpool:
        raise ValueError(
            "Only 'end_maxpool' or 'end_global_maxpool' can be true. Not both.")

    # TODO 1D?
    # Get dimension specific layers
    conv_layer = layers.Conv2D if ndims == 2 else layers.Conv3D
    pool_layer = layers.MaxPool2D if ndims == 2 else layers.MaxPool3D
    global_pool_layer = layers.GlobalMaxPool2D if ndims == 2 else layers.GlobalMaxPool3D

    # Define a block of conv layers
    def conv_block(filt, name, tensor):
        for conv_idx in range(conv_per_block):
            conv_name = name + '_conv' + str(conv_idx)
            tensor = conv_layer(filt, kernel_size,
                                name=conv_name + '_conv')(tensor)
            if batch_norm:
                tensor = layers.BatchNormalization(
                    name=conv_name + '_bn')(tensor)
            tensor = layers.Activation(activation,
                                       name=conv_name + '_' + activation)(tensor)
        return tensor

    def build(tensor):
        for idx, filt in enumerate(filters[:-1]):
            # A block of convs + max pool
            block_name = 'block' + str(idx)
            with K.name_scope(block_name):
                # convs
                tensor = conv_block(filt, block_name, tensor)
                # max pool
                tensor = pool_layer(pool_size=2,
                                    name=block_name + '_pool')(tensor)

        # Last block doesn't always have a max pooling
        block_name = 'block' + str(len(filters) - 1)
        with K.name_scope(block_name):
            tensor = conv_block(filters[-1], block_name, tensor)
            if end_maxpool:
                tensor = pool_layer(pool_size=2)(tensor)
            elif end_global_maxpool:
                tensor = global_pool_layer()(tensor)
        return tensor

    return build
