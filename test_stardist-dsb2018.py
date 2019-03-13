import os

import numpy as np

import matplotlib.pyplot as plt

from keras import layers
from keras import models

from skimage.measure import label

from stardist import dist_to_coord, non_maximum_suppression, polygons_to_label

from keras_transfer_learning.backbones.unet import unet
from keras_transfer_learning.data.stardist_dsb2018 import load_test
from keras_transfer_learning.utils.mean_average_precision import ap_segm
from keras_transfer_learning.heads.stardist import stardist


model_name = 'stardist-small-dsb2018-pretrained'
model_dir = os.path.join('.', 'models', model_name)

n_rays = 32

# Build the model
inp = layers.Input(shape=(None, None, 1))
x = unet([32, 64, 128])(inp)
oups = stardist(n_rays, feature_layer=128)(x)

m = models.Model(inp, oups)

# Print a summary
# m.summary()

# Load the weights
m.load_weights(os.path.join(model_dir, 'weights_final.h5'))

# Run on test data
test_X, test_Y = load_test()

segmentations = []
preds = []
print('Running prediction...')
for x in test_X:
    prob, dist = m.predict(x[None, ..., None])
    prob, dist = prob[0, ..., 0], dist[0]
    coord = dist_to_coord(dist)
    points = non_maximum_suppression(coord, prob, prob_thresh=0.4)
    labels = polygons_to_label(coord, prob, points)
    segmentations.append(labels)

print('Running evaluation...')
ap = ap_segm(segmentations, test_Y, [0.5])

print("The average precision is {}".format(ap))
