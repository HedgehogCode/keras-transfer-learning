import os
import re
from glob import glob
import yaml

import pandas as pd

from keras import layers
from keras import callbacks
from keras import models

from keras_transfer_learning.config import config
from keras_transfer_learning.utils import utils
from keras_transfer_learning.backbones import unet, convnet
from keras_transfer_learning.heads import segm, stardist, classification
from keras_transfer_learning.data import dataaug


###################################################################################################
#     BACKBONE HELPERS
###################################################################################################

def _create_backbone(conf, inp):
    return {
        'unet': lambda: unet.unet(**conf['backbone']['args'])(inp),
        'unet-csbdeep': lambda: unet.unet_csbdeep(**conf['backbone']['args'])(inp),
        'convnet': lambda: convnet.convnet(**conf['backbone']['args'])(inp)
    }[conf['backbone']['name']]()


def _load_weights(conf, model):
    weights = conf['backbone']['weights']
    if weights is not None:
        model.load_weights(weights, by_name=True)


###################################################################################################
#     HEAD HELPERS
###################################################################################################

def _create_head(conf, backbone):
    # FIXME Add unet with border?
    return {
        'fgbg-segm': lambda: segm.segm(num_classes=2, **conf['head']['args'])(backbone),
        'stardist': lambda: stardist.stardist(**conf['head']['args'])(backbone),
        'classification': lambda: classification.classification(**conf['head']['args'])(backbone)
    }[conf['head']['name']]()


def _prepare_model(conf, model):
    return {
        'fgbg-segm': segm.prepare_for_training,
        'stardist': stardist.prepare_for_training,
        'classification': classification.prepare_for_training
    }[conf['head']['name']](model, **conf['head']['prepare_model_args'])


###################################################################################################
#     DATA HELPERS
###################################################################################################

def _create_data_generators(conf):
    # TODO make more general
    # TODO datasplit seed + smaller datasets
    train_x, train_y, val_x, val_y = stardist_dsb2018.load_train(data_dir=conf['data']['data_dir'],
                train_val_split=conf['data']['train_val_split'])

    train_gen = datagen.data_generator_from_lists(batch_size=conf['training']['batch_size'],
                data_x=train_x, data_y=train_y,
                dataaug_fn=_create_dataaug_fn(conf),
                prepare_fn=_create_prepare_fn(conf))


    # FIXME: Create data generators for different datasets
    # load_data()
    # train_generator = conf.data.create_train_datagen(
    #     conf.training.batch_size, conf.head.prepare_data)
    # val_generator = conf.data.create_val_datagen(conf.head.prepare_data)
    return None, None


def _create_dataaug_fn(conf):
    return {
        'imgaug': lambda: dataaug.create_imgaug_augmentor(conf['data']['dataaug']['augmentors']),
    }[conf['data']['dataaug']['name']]()


def _create_prepare_fn(conf):
    # TODO
    return None

###################################################################################################
#     TRAINING HELPERS
###################################################################################################

def _create_callbacks(conf, model_dir):
    training_callbacks = []

    callback_fns = {
        'early_stopping': keras.callbacks.EarlyStopping,
        'reduce_lr_on_plateau': keras.callbacks.ReduceLROnPlateau
    }

    # Default callbackse
    training_callbacks.append(_checkpoints_callback(model_dir))
    training_callbacks.append(_tensorboard_callback(
        conf['name'], conf['training']['batch_size']))

    # Configured callbacks
    for callback in conf['training']['callbacks']:
        training_callbacks.append(
            callback_fns[callback['name']](**callback['args']))

    return training_callbacks


def _checkpoints_callback(model_dir):
    checkpoint_filename = os.path.join(
        model_dir, 'weights_{epoch:04d}_{val_loss:.4f}.h5')
    return callbacks.ModelCheckpoint(
        checkpoint_filename, save_weights_only=True)


def _tensorboard_callback(model_name, batch_size):
    tensorboard_logdir = os.path.join('.', 'logs', model_name)
    return callbacks.TensorBoard(
        tensorboard_logdir, batch_size=batch_size, write_graph=True)


###################################################################################################
#     TRAINING PROCEDURE
###################################################################################################

def train(conf: dict, epochs: int, initial_epoch: int = 0):
    # Get the model directory
    model_dir = os.path.join('.', 'models', conf['name'])

    # Create the input
    inp = layers.Input(conf['input_shape'])

    # Create the backbone
    backbone = _create_backbone(conf, inp)

    # Load pretrained weights
    if initial_epoch == 0:
        backbone_model = models.Model(inputs=inp, outputs=backbone)
        _load_weights(conf, backbone_model)

    # Create the head
    oups = _create_head(conf, backbone)

    # Create the model
    model = models.Model(inputs=inp, outputs=oups)

    # Continue with the training
    if initial_epoch != 0:
        last_weights = utils.get_last_weights(model_dir, initial_epoch)
        model.load_weights(last_weights)

    # Prepare the data generators
    train_generator, val_generator = _create_data_generators(conf)

    # Prepare for training
    model = _prepare_model(conf, model)

    # Create the callbacks
    training_callbacks = _create_callbacks(conf, model_dir)

    # Prepare the model directory
    if initial_epoch == 0:
        if os.path.exists(model_dir):
            raise ValueError(
                "A model with the name {} already exists.".format(conf.name))
        os.makedirs(model_dir)

        # Save the config
        yaml.dump(conf, os.path.join(model_dir, 'config.yaml'))

    # Train the model
    history = model.fit_generator(train_generator, validation_data=val_generator,
                                  epochs=epochs, initial_epoch=initial_epoch,
                                  callbacks=training_callbacks)

    # Save the history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(model_dir, 'history.csv'))

    # Save the final weights
    model.save_weights(os.path.join(model_dir, 'weights_final.h5'))
