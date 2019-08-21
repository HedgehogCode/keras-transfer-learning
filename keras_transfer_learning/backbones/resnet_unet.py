"""ResNet U-Net models for Keras

TODO

"""
from keras import layers

from . import unet, resnet
from .layers import PadToMultiple, CropLike


def resnet_unet(filters=None, blocks=None, ndims=2, padding_fix=True, shortcut=True):
    """Creates a ResNet U-Net. Padding is applied for convolutions. Therefore the input and
    output has the same size.

    Keyword Arguments:
        filters {list of ints} -- list of the filters in the bottleneck layers
                                  (Determines the depth) (default: [32, 64, 128, 256])
        blocks {list of ints or int} -- list of the number of blocks per depth.
        ndims {int} -- number of dimensions (default: {2})
        padding_fix {bool} -- if a padding should be applied to fix the size of the tensor if the
                              size is not divisible by 2 before a max-pooling

    Returns:
        function -- a function which applies a ResNet U-Net to an input tensor
    """
    def _res_conv_block(ndims, filters, blocks=2, name=None):
        return resnet.stack(filters, blocks, stride1=1, ndims=ndims, name=name, shortcut=shortcut)

    if filters is None:
        filters = [32, 64, 128, 256]
    down_filters = filters[:-1]
    middle_filters = filters[-1]
    up_filters = filters[-2::-1]

    depth = len(filters)

    if blocks is None:
        blocks = 2
    if isinstance(blocks, int):
        down_blocks = [blocks] * (depth - 1)
        middle_blocks = blocks
        up_blocks = [blocks] * (depth - 1)
    elif isinstance(blocks, list):
        if len(blocks) == depth:
            down_blocks = blocks[:-1]
            middle_blocks = blocks[-1]
            up_blocks = blocks[-2::-1]
        elif len(blocks) == (depth * 2) - 1:
            down_blocks = blocks[:depth]
            middle_blocks = blocks[depth]
            up_blocks = blocks[depth:]
        else:
            raise ValueError(
                '''Length of blocks must be len(filters) or 2 * len(filters) - 1.
                But is {} while number of filters is {}'''.format(len(blocks), depth))
    else:
        raise ValueError('blocks must be list or int but is {}'.format(type(blocks)))

    def build(tensor):
        # Downsample
        tensors = []
        for idx, (filt, blk) in enumerate(zip(down_filters, down_blocks)):
            tensor = _res_conv_block(
                ndims, filt, blocks=blk, name='features_down' + str(idx))(tensor)
            tensors.insert(0, tensor)
            if padding_fix:
                tensor = PadToMultiple(2)(tensor)
            tensor = unet.downsample_block(ndims,
                                           name='downsample' + str(idx))(tensor)

        # Middle
        tensor = _res_conv_block(
            ndims, middle_filters, blocks=middle_blocks, name='features_middle')(tensor)

        # Upsample
        for idx, (filt, blk) in enumerate(zip(up_filters, up_blocks)):
            tensor = unet.upsample_conv_block(ndims, filt * 4,
                                              name='upsample' + str(idx))(tensor)
            skip_tensor = tensors.pop(0)
            if padding_fix:
                tensor = CropLike()([tensor, skip_tensor])
            tensor = layers.Concatenate(axis=-1,
                                        name='concat' + str(idx))([skip_tensor, tensor])
            tensor = _res_conv_block(
                ndims, filt, blocks=blk, name='features_up' + str(idx))(tensor)
        return tensor
    return build
