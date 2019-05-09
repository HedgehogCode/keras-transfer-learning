"""Custom Keras layers.

PadToMultiple: Pad a tensor such that its spatial dimemsions are a multiple of a given factor
CropLike: Crop a tensor such that its spatial dimensions are the same as another tensor

TODO Also support channels first
"""
import math
import tensorflow as tf
from keras import layers


class PadToMultiple(layers.Layer):

    def __init__(self, factor, **kwargs):
        self.factor = factor
        super(PadToMultiple, self).__init__(**kwargs)

    def build(self, input_shape):
        # No weights to define
        super(PadToMultiple, self).build(input_shape)

    def call(self, t_x):
        ndims = len(t_x.shape) - 2  # - (batch dim + channel dim)
        factors = [1] + ([self.factor] * ndims) + \
            [1]  # Factor 1 for batch + channels

        t_shape = tf.dtypes.cast(
            tf.shape(t_x, out_type=tf.dtypes.int32), dtype=tf.dtypes.float32)
        t_factors = tf.constant(factors, dtype=tf.dtypes.float32)
        t_padded_shape = tf.math.ceil(t_shape / t_factors) * t_factors
        t_paddings = tf.dtypes.cast(
            tf.stack([tf.zeros_like(t_shape), t_padded_shape - t_shape], axis=1),
            dtype=tf.dtypes.int32)
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
        return None if size is None else self.factor * math.ceil(size / self.factor)


class CropLike(layers.Layer):

    def __init__(self, **kwargs):
        super(CropLike, self).__init__(**kwargs)

    def build(self, input_shape):
        self._check_input_shape_arg(input_shape)
        # No weights to define
        super(CropLike, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, list)
        t_x, t_like = x
        ndims = len(t_x.shape) - 2  # - (batch dim + channel dim)
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
