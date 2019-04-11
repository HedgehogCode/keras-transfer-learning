import os
import math

import numpy as np
import tqdm

from keras import layers
from keras import models

import stardist.utils
import stardist.nms

from keras_transfer_learning.utils import utils, mean_average_precision
from keras_transfer_learning.config import config
from keras_transfer_learning.backbones import unet


def evaluate(conf: config.Config):
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
    for x in tqdm.tqdm(test_x):
        # Padding
        pad_0 = 8 * math.ceil(x.shape[0] / 8) - x.shape[0]
        pad_1 = 8 * math.ceil(x.shape[1] / 8) - x.shape[1]
        img = np.pad(x, ((0, pad_0), (0, pad_1)), mode='constant')

        prob, dist = model.predict(img[None, ..., None])

        # Remove padding
        prob = np.take(prob[0, ..., 0], range(0, x.shape[0]), axis=0)
        prob = np.take(prob, range(0, x.shape[1]), axis=1)
        dist = np.take(dist[0], range(0, x.shape[0]), axis=0)
        dist = np.take(dist, range(0, x.shape[1]), axis=1)

        coord = stardist.utils.dist_to_coord(dist)
        points = stardist.nms.non_maximum_suppression(
            coord, prob, prob_thresh=0.4)
        labels = stardist.utils.polygons_to_label(coord, prob, points)
        predictions.append(labels)

    ap = mean_average_precision.ap_segm(predictions, test_y, [0.5])

    print("The average precision is {}".format(ap))
