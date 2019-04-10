import os
import re
from glob import glob

import pandas as pd

from keras import layers
from keras import callbacks
from keras import models

from keras_transfer_learning.config import config
from keras_transfer_learning.utils import utils


def _checkpoints_callback(model_dir):
    checkpoint_filename = os.path.join(
        model_dir, 'weights_{epoch:04d}_{val_loss:.4f}.h5')
    return callbacks.ModelCheckpoint(
        checkpoint_filename, save_weights_only=True)


def _tensorboard_callback(model_name, batch_size):
    tensorboard_logdir = os.path.join('.', 'logs', model_name)
    return callbacks.TensorBoard(
        tensorboard_logdir, batch_size=batch_size, write_graph=True)


def train(conf: config.Config, epochs: int, initial_epoch: int = 0):
    # Get the model directory
    model_dir = os.path.join('.', 'models', conf.name)

    # Create the input
    inp = layers.Input(conf.input_shape)

    # Create the backbone
    backbone = conf.backbone.create_backbone(inp)

    # Load pretrained weights
    if initial_epoch == 0:
        backbone_model = models.Model(inputs=inp, outputs=backbone)
        conf.backbone.load_weights(backbone_model)

    # Create the head
    oups = conf.head.create_head(backbone)

    # Create the model
    model = models.Model(inputs=inp, outputs=oups)

    # Continue with the training
    if initial_epoch != 0:
        last_weights = utils.get_last_weights(model_dir, initial_epoch)
        model.load_weights(last_weights)

    # Prepare the data generators
    conf.data.load_data()
    train_generator = conf.data.create_train_datagen(
        conf.training.batch_size, conf.head.prepare_data)
    val_generator = conf.data.create_val_datagen(conf.head.prepare_data)

    # Prepare for training
    model = conf.head.prepare_model(model)

    # Create the callbacks
    training_callbacks = []
    training_callbacks.append(_checkpoints_callback(model_dir))
    training_callbacks.append(_tensorboard_callback(
        conf.name, conf.training.batch_size))
    training_callbacks.extend(conf.training.create_callbacks())

    # Prepare the model directory
    if initial_epoch == 0:
        if os.path.exists(model_dir):
            raise ValueError(
                "A model with the name {} already exists.".format(conf.name))
        os.makedirs(model_dir)

        # Save the config
        conf.to_yaml(os.path.join(model_dir, 'config.yaml'))


    # Train the model
    history = model.fit_generator(train_generator, validation_data=val_generator,
                                  epochs=epochs, initial_epoch=initial_epoch,
                                  callbacks=training_callbacks)

    # Save the history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(model_dir, 'history.csv'))

    # Save the final weights
    model.save_weights(os.path.join(model_dir, 'weights_final.h5'))
