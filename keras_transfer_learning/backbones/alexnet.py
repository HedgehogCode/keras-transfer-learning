"""AlexNet models for Keras

Code to create simple AlexNet like models.

Paper:
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep
  convolutional neural networks. In Advances in neural information processing systems
  (pp. 1097–1105).

TODO should this really be called alexnet?
"""
import keras.backend as K
from keras import layers


def alexnet(depth=2, conv_per_block=2, kernel_size=3, activation='relu', batch_norm=False, ndims=2):

    def build(tensor):
        # Downsample
        tensors = []
        down_filters = filters[:-1]
        for idx, filt in enumerate(down_filters):
            tensor = conv_block(ndims, filt, kernel_size=kernel_size,
                                activation=activation, batch_norm=batch_norm,
                                name='features_down' + str(idx))(tensor)
            tensors.insert(0, tensor)
            tensor = downsample_block(ndims,
                                      name='downsample' + str(idx))(tensor)

        # Middle
        tensor = conv_block(ndims, filters[-1], kernel_size=kernel_size,
                            activation=activation, batch_norm=batch_norm,
                            name='features_middle')(tensor)

        # Upsample
        up_filters = filters[-2::-1]
        for idx, filt in enumerate(up_filters):
            tensor = upsample_conv_block(ndims, filt,
                                         name='upsample' + str(idx))(tensor)
            tensor = layers.Concatenate(axis=-1,
                                        name='concat' + str(idx))([tensors.pop(0), tensor])
            tensor = conv_block(ndims, filt, kernel_size=kernel_size,
                                activation=activation, batch_norm=batch_norm,
                                name='features_up' + str(idx))(tensor)
        return tensor
    return build
