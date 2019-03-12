import os

import numpy as np

import matplotlib.pyplot as plt

from keras import layers
from keras import models

from skimage.measure import label

from keras_transfer_learning.backbones.unet import unet
from keras_transfer_learning.data.stardist_dsb2018 import loadTest
from keras_transfer_learning.utils.mean_average_precision import ap_segm
from keras_transfer_learning.heads.segm import segm


threshold = 0.5
model_name = 'unet-small-dsb2018'
model_dir = os.path.join('.', 'models', model_name)

# Build the model
inp = layers.Input(shape=(None, None, 1))
x = unet([32, 64, 128])(inp)
oup = segm(2)(x)

m = models.Model(inp, oup)

# Print a summary
# m.summary()

# Load the weights
m.load_weights(os.path.join(model_dir, 'weights_final.h5'))

# Run on test data
test_X, test_Y = loadTest()

segmentations = []
preds = []
print('Running prediction...')
for x in test_X:
    pred = m.predict(x[None, ..., None])[0, ..., 0]
    preds.append(pred)
    segmentations.append(label(np.array(pred > threshold, dtype='uint8')))

print('Running evaluation...')
ap = ap_segm(segmentations, test_Y, [0.5])

print("The average precision is {}".format(ap))