import numpy as np
import tensorflow as tf
from scipy.ndimage import morphology, label, filters
from skimage import measure

import keras.backend as K
from keras import layers, models, losses


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


# =================================================================================================
#     2 CLASS FOREGROUND/BACKGROUND - NOT WEIGHTED
# =================================================================================================

def prepare_data_fgbg(batch_x, batch_y):
    out_x = np.array(batch_x)[..., None]  # TODO input with channels?
    foreground = np.array(batch_y) > 0
    background = np.logical_not(foreground)
    out_y = np.array(
        np.stack([foreground, background], axis=-1), dtype='float32')
    return out_x, out_y


def process_prediction_fgbg(pred, prob_thresh=0.5, do_labeling=True):
    fg_prob = pred[..., 0]

    fg = np.array(fg_prob > prob_thresh, dtype=np.int)

    if not do_labeling:
        return fg

    # Do the labeling
    labels = label(fg)[0]

    # Get the score for each segment by getting the maximum prob in this segment
    score = {}
    for l in np.unique(labels):
        score[l] = np.max(fg_prob[labels == l])

    return labels, score



# =================================================================================================
#     2 CLASS FOREGROUND/BACKGROUND - WEIGHTED
# =================================================================================================

def prepare_for_training_fgbg_weigthed(model, optimizer='adam', loss='binary_crossentropy'):
    # Add the weigth input
    weight_inp = layers.Input(
        model.inputs[0].get_shape().dims[1:-1], name='weight_map')
    m = models.Model(inputs=[model.input, weight_inp], outputs=model.outputs)

    # Get the pixel loss
    pixel_loss = losses.get(loss)

    # Define the weighted loss
    def _weighted_loss(y_true, y_pred):
        return weight_inp * pixel_loss(y_true, y_pred)

    # Compile the model
    m.compile(optimizer, loss=_weighted_loss)
    return m


def prepare_data_fgbg_weigthed(batch_x, batch_y, border_weight=2, separation_border_weight=5, sigma=1):
    # Wrap x in np array and add channel dimension
    out_x = np.array(batch_x)[..., None]  # TODO input with channels?

    # Create the weight map
    struct = morphology.generate_binary_structure(len(batch_y[0].shape), 1)
    foreground = np.zeros_like(batch_y)
    weight_map = np.zeros_like(batch_y, dtype='float32')
    for i, mask in enumerate(batch_y):
        borders = morphology.morphological_laplace(mask, structure=struct) \
            > (np.max(mask) + 1)
        separation_borders = np.logical_and(morphology.grey_erosion(mask, structure=struct),
                                            borders)
        weight_map[i] = separation_border_weight * separation_borders \
            + border_weight * borders \
            + 1

        # Filter weight map
        if sigma > 0:
            weight_map[i] = filters.gaussian_filter(
                weight_map[i], sigma=sigma)

        # Foreground is the mask without the borders
        foreground[i] = np.logical_and((mask > 0), np.logical_not(borders))

    background = np.logical_not(foreground)
    out_y = np.array(
        np.stack([foreground, background], axis=-1), dtype='float32')

    return [out_x, weight_map], out_y


# =================================================================================================
#     N CLASS - NOT WEIGHTED
# =================================================================================================

def prepare_data_nclass(batch_x, batch_y, num_classes):
    out_x = np.array(batch_x)[..., None]  # TODO input with channels?
    batch_y = np.array(batch_y)
    out_y = np.empty((*batch_y.shape, num_classes), dtype='float32')
    np.stack([batch_y == i for i in range(num_classes)], axis=-1, out=out_y)
    return out_x, out_y


def process_prediction_nclass(pred):
    return np.argmax(pred, axis=-1), np.max(pred, axis=-1)


# =================================================================================================
#     WIP MISC
# =================================================================================================

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
