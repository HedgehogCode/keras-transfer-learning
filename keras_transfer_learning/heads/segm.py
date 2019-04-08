import numpy as np
import tensorflow as tf
from scipy.ndimage import morphology
from skimage import measure

import keras.backend as K
from keras import layers


def segm(num_classes, out_activation='softmax', feature_layer=0, feature_kernel_size=3,
         feature_activation='relu', batch_norm=False):
    """Retuns a function that adds a top to a Keras model which outputs a segmentation. The output
    segmentation has the same spatial dimensions as the input tensor.

    To be precise the created function adds a convolutional layer with a 1x1 kernel and an output
    activation to the network. If feature_layer is set to something higher than zero an additional
    convolutional layer is added before the output layer.

    Arguments:
        num_classes {int} -- number of output classes.

    Keyword Arguments:
        out_activation {str} -- activation of the output layer. (default: {'softmax'})
        feature_layer {int} -- number of filters of an additional feature layer. If zero, no
                               feature layer will be added. (default: {0})
        feature_kernel_size {int} -- the kernel size of an additional feature layer. (default: {3})
        feature_activation {str} -- the activation function of an additional feature layer.
                                    (default: {'relu'})
        batch_norm {bool} -- if a batch normalization layer should be added after the feature layer
                             (before the activation). (default: {False})

    Returns:
        function -- a function that adds the top on the given input tensor
    """

    def build(x):
        rank = x.get_shape().ndims
        conv = layers.Conv2D if rank == 4 else layers.Conv3D
        if feature_layer > 0:
            x = conv(feature_layer, feature_kernel_size, padding='same')(x)
            if batch_norm:
                x = layers.BatchNormalization()(x)
            x = layers.Activation(feature_activation)(x)

        x = conv(num_classes, 1, padding='same')(x)
        x = layers.Activation(out_activation)(x)
        return x
    return build


def prepare_for_training(model, optimizer='adam', loss='binary_crossentropy'):
    # Compile the model
    model.compile(optimizer, loss=loss)
    return model


def weighted_crossentropy(y_true, y_pred):
    # TODO check if this is correct
    # TODO not testet!!!!
    [seg, weight] = tf.unstack(y_true, 2, axis=-1)
    fg_bg = K.stack(seg, 1 - seg, -1)

    # Compute the cross-entropy loss
    # TODO check if the orignal paper really uses binary crossentropy
    crossentropy = K.binary_crossentropy(fg_bg, y_pred)

    # Weight pixel-wise crossentropy loss and sum up
    return K.sum(weight * crossentropy)


def unet_weight_map(y, wc=None, w0=10, sigma=5):
    # Copied from https://stackoverflow.com/a/53179982

    labels = measure.label(y)
    no_labels = labels == 0
    label_ids = sorted(np.unique(labels))[1:]

    if len(label_ids) > 1:
        distances = np.zeros((y.shape[0], y.shape[1], len(label_ids)))

        for i, label_id in enumerate(label_ids):
            distances[:, :, i] = morphology.distance_transform_edt(
                labels != label_id)

        distances = np.sort(distances, axis=2)
        d1 = distances[:, :, 0]
        d2 = distances[:, :, 1]
        # TODO check if the weight map is only computed for background pixels
        w = w0 * np.exp(-1/2*((d1 + d2) / sigma)**2) * no_labels

        if wc:
            class_weights = np.zeros_like(y)
            for k, v in wc.items():
                class_weights[y == k] = v
            w = w + class_weights
    else:
        w = np.zeros_like(y)

    return w
