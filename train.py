#!/usr/bin/env python

"""Train a deep learning model according to a configuration file.

TODO enable continuing with training
TODO add seed
"""

import os
import sys
import argparse
import yaml

from keras import layers
from keras import models
from keras import callbacks

from keras_transfer_learning import backbones
from keras_transfer_learning import heads
from keras_transfer_learning import data


# Decisions based on backbone, head and data

BACKBONE_FNS = {
    'unet': backbones.unet.unet
}

HEAD_FNS = {
    'segm': heads.segm.segm
}

PREPARE_MODEL_FNS = {
    'segm': heads.segm.prepare_for_training
}

CALLBACK_FNS = {
    'early_stopping': callbacks.EarlyStopping,
    'reduce_lr_on_plateau': callbacks.ReduceLROnPlateau
}

LOAD_DATA_FNS = {
    'stardist-dsb2018': data.stardist_dsb2018.load_train
}

PREPARE_DATA_FNS = {
    'segm': heads.segm.prepare_data_fn
}

GENERATOR_FNS = {
    'stardist-dsb2018': (train_gen_from_lists, val_gen_from_lists)
}


# Helper functions

def checkpoints_callback(model_dir):
    checkpoint_filename = os.path.join(
        model_dir, 'weights_{epoch:02d}_{val_loss:.2f}.h5')
    return callbacks.ModelCheckpoint(
        checkpoint_filename, save_weights_only=True)


def tensorboard_callback(model_name, batch_size):
    tensorboard_logdir = os.path.join('.', 'logs', model_name)
    return callbacks.TensorBoard(
        tensorboard_logdir, batch_size=batch_size, write_graph=True)


def train_gen_from_lists(dataset, batch_size, dataaug_fn, prepare_fn, seed):
    return data.datagen.data_generator_from_lists(batch_size=batch_size,
                                                  data_x=dataset[0], data_y=dataset[1],
                                                  dataaug_fn=dataaug_fn,
                                                  prepare_fn=prepare_fn,
                                                  seed=seed)


def val_gen_from_lists(dataset, prepare_fn):
    return data.datagen.data_generator_for_validation(val_x=dataset[2],
                                                      val_y=dataset[3],
                                                      prepare_fn=prepare_fn)


# Main function

def main(args):
    config = yaml.load(args.configfile)
    epochs = args.epochs
    print(config)
    print(epochs)

    # Split up the config
    config_name = config['name']
    config_backbone = config['backbone']
    config_head = config['head']
    config_data = config['data']
    config_training = config['training']

    # Prepare the model directory
    model_dir = os.path.join('.', 'models', config_name)
    # TODO uncomment
    # if os.path.exists(model_dir):
    #     raise ValueError(
    #         "A model with the name {} already exists.".format(config_name))
    # os.makedirs(model_dir)

    # Create the input
    inp = layers.Input(config['input_shape'])

    # Create the backbone
    backbone = BACKBONE_FNS[config_backbone['type']](
        **config_backbone['args'])(inp)

    # TODO load pretrained weights

    # Create the head
    oups = HEAD_FNS[config_head['type']](**config_head['args'])(backbone)

    # Create the model
    model = models.Model(inputs=inp, outputs=oups)

    # TODO Prepare the data generators
    dataset = LOAD_DATA_FNS[config_data['name']](config_data['load_args'])
    generator_fns = GENERATOR_FNS[config_data['name']]
    dataaug_fn = data.datagen.dataug_fn_crop_flip_2d(
        128, 128)  # TODO make configurable
    prepare_fn = PREPARE_DATA_FNS[config_head['name']](
        config_head['prepare_data_args'])
    train_generator = generator_fns[0](
        dataset, config_training['batch_size'], dataaug_fn, prepare_fn, config_data['seed'])
    val_generator = generator_fns[1](dataset, prepare_fn)
    # TODO CONTINUE HERE

    # Prepare for training
    # TODO should these functions also return a function?
    model = PREPARE_MODEL_FNS[config_head['type']](
        model, **config_head['prepare_model_args'])

    # Create the callbacks
    training_callbacks = []
    training_callbacks.append(checkpoints_callback(model_dir))
    training_callbacks.append(tensorboard_callback(
        config_name, config_training['batch_size']))
    for callback in config_training['callbacks']:
        training_callbacks.append(
            CALLBACK_FNS[callback['name']](**callback['args']))

    # Train the model
    history = model.fit_generator(train_generator, epochs=epochs,
                                  callbacks=training_callbacks, validation_data=val_generator)

    # TODO Save the history
    print(history)

    # Save the final weights
    model.save_weights(os.path.join(model_dir, 'weights_final.h5'))


# Argument parsing

def parse_args(arguments):
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('configfile', help='Config file',
                        type=argparse.FileType('r'))
    parser.add_argument(
        '-e', '--epochs', help='Number of epochs', type=int, required=True)
    # TODO add other arguments to control training
    # Especially continue training

    return parser.parse_args(arguments)


if __name__ == '__main__':
    sys.exit(main(parse_args(sys.argv[1:])))
