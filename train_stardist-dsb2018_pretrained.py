import os

import numpy as np

from keras import layers
from keras import models
from keras import callbacks

from stardist.utils import edt_prob, star_dist

from keras_transfer_learning.backbones.unet import unet
from keras_transfer_learning.data.datagen import data_generator_from_lists
from keras_transfer_learning.data.datagen import data_generator_for_validation
from keras_transfer_learning.data.datagen import dataaug_fn_crop_flip_2d
from keras_transfer_learning.data.stardist_dsb2018 import load_train
from keras_transfer_learning.heads.stardist import stardist, prepare_for_training

model_name = 'stardist-small-dsb2018-pretrained'
model_dir = os.path.join('.', 'models', model_name)
os.makedirs(model_dir, exist_ok=True)

n_rays = 32

# Build the model
inp = layers.Input(shape=(None, None, 1))
x = unet([32, 64, 128])(inp)

# Load the weights of the other model
backbone = models.Model(inp, x)
backbone.load_weights(os.path.join(
    '.', 'models', 'unet-small-dsb2018', 'weights_final.h5'), by_name=True)

# Add the stardist head
oups = stardist(n_rays, feature_layer=128)(x)

# Create the model
m = models.Model(inp, oups)

# Prepare the data
train_x, train_y, val_x, val_y = load_train()

# TODO move to stardist module?
def prepare_fn(batch_x, batch_y):
    prob = np.stack([edt_prob(lbl) for lbl in batch_y])[...,None]
    dist = np.stack([star_dist(lbl, n_rays) for lbl in batch_y])
    dist_mask = prob
    img = (np.array(batch_x, dtype='float32') / 255)[..., None]
    return [img, dist_mask], [prob, dist]


dataaug_fn = dataaug_fn_crop_flip_2d(128, 128)

batch_size = 8
epochs = 10

train_generator = data_generator_from_lists(
    batch_size=batch_size, data_x=train_x, data_y=train_y, dataaug_fn=dataaug_fn, prepare_fn=prepare_fn)
val_generator = data_generator_for_validation(val_x, val_y, prepare_fn)

# Prepare the model
m = prepare_for_training(m)

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
