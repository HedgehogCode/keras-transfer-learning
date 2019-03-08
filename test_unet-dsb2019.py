import os

import numpy as np

import matplotlib.pyplot as plt

from keras import layers
from keras import models

from skimage.measure import label
from scipy.ndimage import find_objects

from keras_transfer_learning.backbones.unet import unet
from keras_transfer_learning.data.stardist_dsb2018 import loadTest


threshold = 0.5
model_name = 'unet-small-dsb2018'
model_dir = os.path.join('.', 'models', model_name)

# Build the model
inp = layers.Input(shape=(None, None, 1))
x = unet([32, 64, 128])(inp)
x = layers.Conv2D(1, (1, 1))(x)
oup = layers.Activation('sigmoid')(x)

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

# Compute mean average precision
def overlaps_1d(first, second, size):
    """Computes if slice first and second overlap for the size of the dimension"""
    first_min, first_max, _ = first.indices(size)
    second_min, second_max, _ = second.indices(size)
    return first_max >= second_min and second_max >= first_min

def overlaps(firsts, seconds, shape):
    """Computes if slices firsts and seconds all overlap for the given shape"""
    for first, second, size in zip(firsts, seconds, shape):
        if not overlaps_1d(first, second, size):
            return False
    return True

print('Running evaluation...')
segments = []
for i, (pred, gt) in enumerate(zip(segmentations, test_Y)):
    pred_segments = find_objects(pred)
    gt_segments = find_objects(gt)

    for pred_segment in pred_segments:
        best_gt = None
        best_gt_iou = 0
        for gt_segment in gt_segments:
            if overlaps(pred_segment, gt_segment, shape)



