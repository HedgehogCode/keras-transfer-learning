"""General training script. Uses a config dictionary to create the model and train it on configured
data.

LIST OF TODOS:
TODO Fix unet image size limitation
TODO Add unet with border (head)
"""
import os
import yaml

import pandas as pd

from keras import layers
from keras import callbacks
from keras import models

from keras_transfer_learning.utils import utils
from keras_transfer_learning.backbones import unet, convnet
from keras_transfer_learning.heads import segm, stardist, classification
from keras_transfer_learning.data import dataaug, datagen, stardist_dsb2018, cytogen


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
    # Decide which function is appropriate
    return {
        'stardist_dsb2018': _create_data_generators_from_lists,
        'cytogen': _create_data_generators_from_lists
    }[conf['data']['name']](conf)


def _create_data_generators_from_lists(conf):
    # Find the appropriate load function
    load_fn = {
        'stardist_dsb2018': stardist_dsb2018.load_train,
        'cytogen': cytogen.load_train
    }[conf['data']['name']]

    # Load the data
    seed = conf['data'].get('datasplit_seed', 42)
    train_val_split = conf['data'].get('train_val_split', 0.9)
    num_train = conf['data'].get('num_train', None)
    part = conf['data'].get('part', 0)
    train_x, train_y, val_x, val_y = load_fn(
        data_dir=conf['data']['data_dir'], seed=seed, train_val_split=train_val_split,
        num_train=num_train, part=part)

    # Create the prepare and dataaug functions
    prepare_fn = _create_prepare_fn(conf)
    dataaug_fn = _create_dataaug_fn(conf)

    # Create the generators
    train_gen = datagen.data_generator_from_lists(
        batch_size=conf['training']['batch_size'], data_x=train_x, data_y=train_y,
        dataaug_fn=dataaug_fn, prepare_fn=prepare_fn)
    val_gen = datagen.data_generator_for_validation(
        val_x=val_x, val_y=val_y, prepare_fn=prepare_fn)

    return train_gen, val_gen


def _create_dataaug_fn(conf):
    return {
        'imgaug': lambda: dataaug.create_imgaug_augmentor(conf['data']['dataaug']['augmentors']),
    }[conf['data']['dataaug']['name']]()


def _create_prepare_fn(conf):
    return {
        'stardist': lambda x, y: stardist.prepare_data(conf['head']['args']['n_rays'], x, y),
        'fgbg-segm': segm.prepare_data_fgbg,
        'classification': classification.prepare_data
    }[conf['head']['name']]


###################################################################################################
#     TRAINING HELPERS
###################################################################################################

def _create_callbacks(conf, model_dir):
    training_callbacks = []

    callback_fns = {
        'early_stopping': callbacks.EarlyStopping,
        'reduce_lr_on_plateau': callbacks.ReduceLROnPlateau
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
