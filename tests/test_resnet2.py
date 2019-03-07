import keras.backend as K
from keras.layers import Input
from keras_transfer_learning.backbones.resnet2 import resnet2_50, resnet2_101, resnet2_152


# ImageNet shape

def test_resnet2_50_imagenet_shape():
    K.clear_session()
    x = Input(shape=(224, 224, 3))
    m = resnet2_50(x)


def test_resnet2_101_imagenet_shape():
    K.clear_session()
    x = Input(shape=(224, 224, 3))
    m = resnet2_101(x)


def test_resnet2_152_imagenet_shape():
    K.clear_session()
    x = Input(shape=(224, 224, 3))
    m = resnet2_152(x)


# Other shape

def test_resnet2_50_other_shape():
    K.clear_session()
    x = Input(shape=(111, 113, 22))
    m = resnet2_50(x)


def test_resnet2_101_other_shape():
    K.clear_session()
    x = Input(shape=(111, 113, 22))
    m = resnet2_101(x)


def test_resnet2_152_other_shape():
    K.clear_session()
    x = Input(shape=(111, 113, 22))
    m = resnet2_152(x)


# No shape

def test_resnet2_50_none_shape():
    K.clear_session()
    x = Input(shape=(None, None, 4))
    m = resnet2_50(x)


def test_resnet2_101_none_shape():
    K.clear_session()
    x = Input(shape=(None, None, 4))
    m = resnet2_101(x)


def test_resnet2_152_none_shape():
    K.clear_session()
    x = Input(shape=(None, None, 4))
    m = resnet2_152(x)
