import tensorflow as tf

from keras import layers
from keras import models

from stardist.model import masked_loss_mae, masked_loss_mse


def stardist(n_rays, feature_layer=0, feature_kernel_size=3, feature_activation='relu',
             batch_norm=False):
    """Retuns a function that adds a top to a Keras model which outputs a the distances and
    probabilities of a StarDist model. The output has the same spatial dimensions as the input
    tensor.

    TODO describe better

    Arguments:
        n_rays {int} -- number of rays of the predicted star-shaped polygon

    Keyword Arguments:
        feature_layer {int} -- number of filters of an additional feature layer. If zero, no
                               feature layer will be added. (default: {0})
        feature_kernel_size {int} -- the kernel size of an additional feature layer. (default: {3})
        feature_activation {str} -- the activation function of an additional feature layer.
                                    (default: {'relu'})
        batch_norm {bool} -- if a batch normalization layer should be added after the feature layer
                             (before the activation). (default: {False})

    Returns:
        function -- a function that adds the top on the given input tensor and returns the prob
                    output and the dist output in a tuple
    """
    def build(x):
        if feature_layer > 0:
            x = layers.Conv2D(
                feature_layer, feature_kernel_size, padding='same')(x)
            if batch_norm:
                x = layers.BatchNormalization()(x)
            x = layers.Activation(feature_activation)(x)

        oup_prob = layers.Conv2D(
            1, (1, 1), padding='same', activation='sigmoid', name='prob')(x)
        oup_dist = layers.Conv2D(
            n_rays, (1, 1), padding='same', activation='linear', name='dist')(x)
        return oup_prob, oup_dist
    return build


def prepare_for_training(model, optimizer='adam', dist_loss='mae'):
    # Add the mask input
    inp_mask = layers.Input(
        model.outputs[0].get_shape().dims[1:], name='dist_mask')
    m = models.Model([model.input, inp_mask], model.outputs)

    # Prepare the dist_loss
    if dist_loss == 'mae':
        dist_loss_fn = masked_loss_mae(inp_mask)
    elif dist_loss == 'mse':
        dist_loss_fn = masked_loss_mse(inp_mask)
    else:
        raise ValueError(
            "The dist loss must be either 'mse' or 'mae' but is {}.".format(dist_loss))

    # Compile the model
    m.compile(optimizer, loss=['binary_crossentropy', dist_loss_fn])
    return m
