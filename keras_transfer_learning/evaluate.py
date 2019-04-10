import os

from keras import layers
from keras import models

import stardist.utils
import stardist.nms

from keras_transfer_learning.utils import utils, mean_average_precision
from keras_transfer_learning import config


def evaluate(conf: config.config.Config):
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

    # TODO the following is only for the stardist model for now. This needs to be generalized
    test_x, test_y = conf.data.create_test_dataset()

    predictions = []
    for x in test_x:
        prob, dist = model.predict(x[None, ..., None])
        coord = stardist.utils.dist_to_coord(dist)
        points = stardist.nms.non_maximum_suppression(
            coord, prob, prob_thresh=0.4)
        labels = stardist.utils.polygons_to_label(coord, prob, points)
        predictions.append(labels)

    ap = mean_average_precision.ap_segm(predictions, test_y, [0.5])

    print("The average precision is {}".format(ap))
