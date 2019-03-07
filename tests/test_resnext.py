import keras.backend as K
from keras.layers import Input
from keras_transfer_learning.backbones.resnext import resnext_50, resnext_101


# ImageNet shape

def test_resnext_50_imagenet_shape():
    K.clear_session()
    x = Input(shape=(224, 224, 3))
    m = resnext_50(x)


def test_resnext_101_imagenet_shape():
    K.clear_session()
    x = Input(shape=(224, 224, 3))
    m = resnext_101(x)


# Other shape

def test_resnext_50_other_shape():
    K.clear_session()
    x = Input(shape=(111, 113, 22))
    m = resnext_50(x)


def test_resnext_101_other_shape():
    K.clear_session()
    x = Input(shape=(111, 113, 22))
    m = resnext_101(x)
