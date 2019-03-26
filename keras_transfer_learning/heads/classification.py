import tensorflow as tf

import keras.backend as K
from keras import layers


def classification(n_classes: int, flatten: bool = True, feature_layer: int = 0,
                   feature_activation='relu', dropout: float = 0., batch_norm: bool = False):
    """Retuns a function that adds a top to a Keras model which outputs a classification.

    TODO describe better
    TODO does something else then softmax make sense as activation function

    Arguments:
        n_classes {int} -- number of classes to predict

    Keyword Arguments:
        flatten {bool} -- if a flatten layer should be added. This should be true if the previous
                          layer has more than one dimension. (default: {True})
        feature_layer {int} -- number of neurons of an additional dense feature layer. If zero, no
                               feature layer will be added. (default: {0})
        feature_activation {str} -- the activation function of an additional feature layer.
                                    (default: {'relu'})
        dropout {float} -- the dropout ratio. If zero, no dropout layer will be added. Only applies
                           if a feature layer was added. (default: {0})
        batch_norm {bool} -- if a batch normalization layer should be added after the feature layer
                             (before the activation). (default: {False})

    Returns:
        function -- a function that adds the top on the given input tensor and returns the
                    classification probability output
    """
    def build(x):
        with K.name_scope('head'):
            if flatten:
                x = layers.Flatten(name='head_flatten')(x)

            if feature_layer > 0:
                x = layers.Dense(feature_layer, name='head_features')(x)
                if dropout > 0:
                    x = layers.Dropout(dropout, name='head_dropout')(x)
                if batch_norm:
                    x = layers.BatchNormalization(name='head_batchnorm')(x)
                x = layers.Activation(feature_activation,
                                      name='head_activation')(x)

            oup = layers.Dense(n_classes, activation='softmax')(x)
        return oup
    return build


def prepare_for_training(model, optimizer='adam', loss='categorical_crossentropy'):
    # Compile the model
    m.compile(optimizer, loss=loss)
    return m
