import os

import numpy as np

from keras import layers
from keras import models
from keras import callbacks

from keras_transfer_learning.backbones.unet import unet
from keras_transfer_learning.data.datagen import data_generator_from_lists
from keras_transfer_learning.data.datagen import data_generator_for_validation
from keras_transfer_learning.data.datagen import dataug_fn_crop_flip_2d
from keras_transfer_learning.data.stardist_dsb2018 import loadTrain

model_name = 'unet-small-dsb2018'
model_dir = os.path.join('.', 'models', model_name)
os.makedirs(model_dir, exist_ok=True)

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

batch_size = 8
epochs = 10

train_generator = data_generator_from_lists(
    batch_size=batch_size, data_x=train_X, data_y=train_Y, dataaug_fn=dataaug_fn, prepare_fn=prepare_fn)
val_generator = data_generator_for_validation(val_X, val_Y, prepare_fn)

# Prepare the model
m.compile('adam', 'binary_crossentropy')

# Train the model
checkpoint_filename = os.path.join(
    model_dir, 'weights_{epoch:02d}_{val_loss:.2f}.h5')
checkpoints = callbacks.ModelCheckpoint(
    checkpoint_filename, save_weights_only=True)

tensorboard_logdir = os.path.join('.', 'logs', model_name)
tensorboard = callbacks.TensorBoard(
    tensorboard_logdir, batch_size=batch_size, write_graph=True)

m.fit_generator(train_generator, epochs=epochs,
                callbacks=[checkpoints, tensorboard], validation_data=val_generator)

m.save_weights(os.path.join(model_dir, 'weights_final.h5'))
