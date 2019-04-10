import os

from keras import layers
from keras import models

from keras_transfer_learning.utils import utils


def evaluate(conf):
    # Get the model directory
    model_dir = os.path.join('.', 'models', conf.name)

    # Create the model
    inp = layers.Input(conf.input_shape)
    backbone = conf.backbone.create_backbone(inp)
    oups = conf.head.create_head(backbone)
    model = models.Model(inputs=inp, outputs=oups)

    # Load the weights
    # TODO allow to run the evalutation on muliple weights files (different epochs)
    last_weights = utils.get_last_weights(model_dir)
    model.load_weights(last_weights)

    # TODO load the data
    # Maybe a data generator like the validataion generator?

    # TODO predict

    # TODO prediction to segmentation/instance segm/...

    # TODO evaluate
