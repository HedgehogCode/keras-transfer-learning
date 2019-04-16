"""U-Net models for Keras

The original U-Net is implemented with slight modifications in the function `unet`.
Paper:
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical
  image segmentation. In International Conference on Medical image computing and computer-assisted
  intervention (pp. 234–241).

The U-Net used in CSBDeep and StarDist is implemented in the function `unet_csbdeep`.
Papers:
- Weigert, M., Schmidt, U., Boothe, T., Müller, A., Dibrov, A., Jain, A., … Myers, E. W. (2018).
  Content-aware image restoration: pushing the limits of fluorescence microscopy. Nature Methods,
  15(12), 1090–1097. https://doi.org/10.1038/s41592-018-0216-7
- Schmidt, U., Weigert, M., Broaddus, C., & Myers, G. (2018). Cell Detection with Star-convex
  Polygons. ArXiv Preprint ArXiv:1806.03535.

TODO valid paddding
"""
import math

import numpy as np
import tensorflow as tf

import keras.backend as K
from keras import layers


def unet(filters=None, kernel_size=3, activation='relu', batch_norm=False, ndims=2,
         padding_fix=False):
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
        padding_fix {bool} -- if a padding should be applied to fix the size of the tensor if the
                              size is not divisible by 2 before a max-pooling

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
            if padding_fix:
                tensor = _PadToMultiple(2)(tensor)
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
            skip_tensor = tensors.pop(0)
            if padding_fix:
                _CropLike()([tensor, skip_tensor])
            tensor = layers.Concatenate(axis=-1,
                                        name='concat' + str(idx))([skip_tensor, tensor])
            tensor = conv_block(ndims, filt, kernel_size=kernel_size,
                                activation=activation, batch_norm=batch_norm,
                                name='features_up' + str(idx))(tensor)
        return tensor
    return build


def unet_csbdeep(filter_base=32, depth=3, conv_per_depth=2, kernel_size=3, activation='relu',
                 batch_norm=False, ndims=2, padding_fix=False):
    """Creates a U-Net like the U-Nets used in the CSBDeep/CARE paper and StarDist paper.

    Keyword Arguments:
        filter_base {int} -- number of the filters on the first level (default: {32})
        depth {int} -- depth of the U-Net (how many downsampling blocks will be created)
                       (default: {3})
        conv_per_depth {int} -- number of convolutional layers per block (default: {2})
        kernel_size {int} -- kernel size of the convolutional layers (default: {3})
        activation {str} -- activation function after the convolutional layers (default: {'relu'})
        batch_norm {bool} -- if batch normalization should be added (default: {False})
        ndims {int} -- number of dimensions (default: {2})
        padding_fix {bool} -- if a padding should be applied to fix the size of the tensor if the
                              size is not divisible by 2 before a max-pooling

    Returns:
        function -- a function which applies a U-Net to an input tensor
    """
    def _upsample_filters(idx):
        filt = [filter_base * 2 ** idx] * conv_per_depth
        filt[-1] = int(filt[-1] / 2)
        return filt

    def build(tensor):
        # Downsample
        tensors = []
        for idx in range(depth):
            filt = filter_base * 2 ** idx
            tensor = conv_block(ndims, filt, num=conv_per_depth, kernel_size=kernel_size,
                                activation=activation, batch_norm=batch_norm,
                                name='features_down' + str(idx))(tensor)
            tensors.insert(0, tensor)
            if padding_fix:
                tensor = _PadToMultiple(2)(tensor)
            tensor = downsample_block(ndims,
                                      name='downsample' + str(idx))(tensor)

        # Middle
        filt = _upsample_filters(depth)
        tensor = conv_block(ndims, filt, num=conv_per_depth, kernel_size=kernel_size,
                            activation=activation, batch_norm=batch_norm,
                            name='features_middle')(tensor)

        # Upsample
        for idx in range(depth):
            filt = _upsample_filters(idx)
            tensor = upsample_block(ndims,
                                    name='upsample' + str(idx))(tensor)
            skip_tensor = tensors.pop(0)
            if padding_fix:
                _CropLike()([tensor, skip_tensor])
            tensor = layers.Concatenate(axis=-1,
                                        name='concat' + str(idx))([skip_tensor, tensor])
            tensor = conv_block(ndims, filt, num=conv_per_depth, kernel_size=kernel_size,
                                activation=activation, batch_norm=batch_norm,
                                name='features_up' + str(idx))(tensor)
        return tensor
    return build


def conv_block(ndims, filters, num=2, kernel_size=3, padding='same', activation='relu',
               batch_norm=False, name=None):
    """Creates a block of two convolutional layers with actitivations after the convolutions.

    Arguments:
        ndims {int} -- number of dimensions
        filters {int or list} -- number of filters for both convolutional layers.
        num {int} -- number of convolutional layers

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

    if isinstance(filters, list):
        assert len(filters) == num
    if isinstance(filters, int):
        filters = [filters] * num

    def build(tensor):
        with K.name_scope(name):
            for idx, filt in enumerate(filters, 1):
                tensor = conv_fn(filt, kernel_size, padding=padding,
                                 name=name + '_conv' + str(idx))(tensor)
                if batch_norm:
                    tensor = layers.BatchNormalization(
                        name=name + '_bn' + str(idx))(tensor)
                tensor = layers.Activation(activation,
                                           name=name + '_' + activation + str(idx))(tensor)
        return tensor
    return build


def upsample_conv_block(ndims, filters, size=2, padding='same', name=None):
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


def upsample_block(ndims, size=2, name=None):
    """Creates an upsample block which consists of a upsampling layer.

    Arguments:
        ndims {int} -- number of dimensions

    Keyword Arguments:
        size {tuple} -- size of the upsampling and convolutional kernels (default: {(2, 2, 2)})
        name {[type]} -- name of the block (default: {None})

    Returns:
        function -- a function which applies this block to a tensor
    """
    upsampling_fn = layers.UpSampling2D if ndims == 2 else layers.UpSampling3D

    def build(tensor):
        with K.name_scope(name):
            tensor = upsampling_fn(size, name=name + '_upsample')(tensor)
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


def crop_data(data, factor: int = 8):
    """Crops the numpy data arrays such that the spatial dimensions (1 and 2) are a multiple of
    the factor argument.

    Arguments:
        data {ndarray} -- the data with the dimensions (batch,width,height,...)

    Keyword Arguments:
        factor {int} -- each spatial dimension is a muliple of factor after the crop

    Returns:
        ndarray -- the cropped data
    """
    cropped = data
    cropped = _crop_axis(cropped, factor, 1)
    cropped = _crop_axis(cropped, factor, 2)
    return cropped


def _crop_axis(data, factor: int, axis: int):
    size = data.shape[axis]
    cropped_size = factor * (size // factor)
    return np.take(data, range(cropped_size), axis=axis)


# TODO Also support channels first
class _PadToMultiple(layers.Layer):

    def __init__(self, factor, **kwargs):
        self.factor = factor
        super(_PadToMultiple, self).__init__(**kwargs)

    def build(self, input_shape):
        # No weights to define
        super(_PadToMultiple, self).build(input_shape)

    def call(self, t_x):
        ndims = t_x.numDimensions() - 2  # - (batch dim + channel dim)
        factors = [1] + ([self.factor] * ndims) + \
            [1]  # Factor 1 for batch + channels

        t_shape = tf.shape(t_x, out_type=tf.dtypes.int32)
        t_factors = tf.constant(factors, dtype=tf.dtypes.float32)
        t_padded_shape = tf.math.ceil(t_shape / t_factors) * t_factors
        t_paddings = tf.stack(
            [tf.zeros_like(t_shape), t_padded_shape - t_shape], axis=0)
        return tf.pad(t_x, t_paddings)

    def compute_output_shape(self, input_shape):
        ndims = len(input_shape) - 1  # - channel dim
        if ndims == 1:
            return (self._pad_axis(input_shape[0]), input_shape[1])
        if ndims == 2:
            return (self._pad_axis(input_shape[0]), self._pad_axis(input_shape[1]), input_shape[2])
        if ndims == 3:
            return (self._pad_axis(input_shape[0]), self._pad_axis(input_shape[1]),
                    self._pad_axis(input_shape[2]), input_shape[3])
        raise ValueError(
            'Must be a 1D, 2D or 3D input. Was {}D.'.format(ndims))

    def _pad_axis(self, size):
        return self.factor * math.ceil(size / self.factor)


class _CropLike(layers.Layer):

    def __init__(self, **kwargs):
        super(_CropLike, self).__init__(**kwargs)

    def build(self, input_shape):
        self._check_input_shape_arg(input_shape)
        # No weights to define
        super(_CropLike, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, list)
        t_x, t_like = x
        ndims = t_x.numDimensions() - 2  # - (batch dim + channel dim)
        t_like_shape = tf.shape(t_like)
        if ndims == 1:
            return t_x[:, :t_like_shape[1], :]
        if ndims == 2:
            return t_x[:, :t_like_shape[1], :t_like_shape[2], :]
        if ndims == 3:
            return t_x[:, :t_like_shape[1], :t_like_shape[2], :t_like_shape[3], :]
        raise ValueError(
            'Must be a 1D, 2D or 3D input. Was {}D.'.format(ndims))

    def compute_output_shape(self, input_shape):
        self._check_input_shape_arg(input_shape)
        x_shape, like_shape = input_shape
        ndims = len(x_shape) - 1  # - channel dim
        if ndims == 1:
            return (like_shape[0], x_shape[1])
        if ndims == 2:
            return (like_shape[0], like_shape[1], x_shape[2])
        if ndims == 3:
            return (like_shape[0], like_shape[1], like_shape[2], x_shape[3])
        raise ValueError(
            'Must be a 1D, 2D or 3D input. Was {}D.'.format(ndims))

    def _check_input_shape_arg(self, input_shape):
        assert isinstance(input_shape, list)
        assert len(input_shape[0]) == len(input_shape[1])
