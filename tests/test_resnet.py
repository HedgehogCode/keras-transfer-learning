import keras.backend as K
from keras.layers import Input
from keras_transfer_learning.backbones.resnet import resnet_38, resnet_50, resnet_101, resnet_152


# ImageNet shape

def test_resnet_38_imagenet_shape():
    K.clear_session()
    x = Input(shape=(224, 224, 3))
    m = resnet_38(x)


def test_resnet_50_imagenet_shape():
    K.clear_session()
    x = Input(shape=(224, 224, 3))
    m = resnet_50(x)


def test_resnet_101_imagenet_shape():
    K.clear_session()
    x = Input(shape=(224, 224, 3))
    m = resnet_101(x)


def test_resnet_152_imagenet_shape():
    K.clear_session()
    x = Input(shape=(224, 224, 3))
    m = resnet_152(x)


# Other shape

def test_resnet_38_other_shape():
    K.clear_session()
    x = Input(shape=(111, 113, 22))
    m = resnet_38(x)


def test_resnet_50_other_shape():
    K.clear_session()
    x = Input(shape=(111, 113, 22))
    m = resnet_50(x)


def test_resnet_101_other_shape():
    K.clear_session()
    x = Input(shape=(111, 113, 22))
    m = resnet_101(x)


def test_resnet_152_other_shape():
    K.clear_session()
    x = Input(shape=(111, 113, 22))
    m = resnet_152(x)


# No shape

def test_resnet_38_none_shape():
    K.clear_session()
    x = Input(shape=(None, None, 4))
    m = resnet_38(x)


def test_resnet_50_none_shape():
    K.clear_session()
    x = Input(shape=(None, None, 4))
    m = resnet_50(x)


def test_resnet_101_none_shape():
    K.clear_session()
    x = Input(shape=(None, None, 4))
    m = resnet_101(x)


def test_resnet_152_none_shape():
    K.clear_session()
    x = Input(shape=(None, None, 4))
    m = resnet_152(x)
