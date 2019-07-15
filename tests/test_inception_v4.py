from keras import layers, models, utils

from keras_transfer_learning import backbones


def test_inception_v4_load_weights():
    inp = layers.Input((None, None, 3))
    oup = backbones.inception_v4.inception_v4_base(inp)

    backbone_model = models.Model(inp, oup)

    weights_path = utils.get_file(
        'inception-v4_weights_tf_dim_ordering_tf_kernels_notop.h5',
        backbones.inception_v4.WEIGHTS_PATH_NO_TOP,
        cache_subdir='models',
        md5_hash='9296b46b5971573064d12e4669110969')

    backbone_model.load_weights(weights_path, by_name=True)

def test_inception_v4_load_weights():
    inp = layers.Input((512, 512, 3))
    oup = backbones.inception_v4.inception_v4_base(inp, mode='unet')

    backbone_model = models.Model(inp, oup)
