import os

import numpy as np

from keras import layers
from keras import models
import keras.backend as K

from keras_transfer_learning.backbones.unet import unet
from keras_transfer_learning.data.data_generator import data_generator_from_lists, dataug_fn_crop_flip_2d
from keras_transfer_learning.data.stardist_dsb2018 import loadTrain

# Build the model
inp = layers.Input(shape=(None, None, 1))
x = unet([32, 64, 128])(inp)
x = layers.Conv2D(1, (1, 1))(x)
oup = layers.Activation('sigmoid')(x)

m = models.Model(inp, oup)

# Print a summary
# m.summary()

# Prepare the data

train_X, train_Y, val_X, val_Y = loadTrain()


def prepare_fn(X, Y):
    X = (np.array(X, dtype='float32') / 255)[..., None]
    Y = (np.array(np.array(Y) > 0, dtype='float32'))[..., None]
    return X, Y


dataaug_fn = dataug_fn_crop_flip_2d(128, 128)

train_generator = data_generator_from_lists(
    batch_size=8, data_X=train_X, data_Y=train_Y, dataaug_fn=dataaug_fn, prepare_fn=prepare_fn)

# Prepare the model
m.compile('adam', 'binary_crossentropy')

# Train the model
m.fit_generator(train_generator, epochs=10)
