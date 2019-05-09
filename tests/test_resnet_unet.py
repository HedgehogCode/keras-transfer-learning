import keras.backend as K
from keras.layers import Input
from keras_transfer_learning.backbones.resnet_unet import resnet_unet


def test_resnet_unet_default():
    K.clear_session()
    x = Input(shape=(None, None, 1))
    m = resnet_unet()(x)


def test_resnet_unet_deep():
    K.clear_session()
    x = Input(shape=(111, 113, 22))
    m = resnet_unet(filters=[16, 32, 64, 128, 256, 512],
                    blocks=[2, 3, 3, 3, 4, 4, 4, 3, 3, 2, 2])(x)
