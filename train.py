#!/usr/bin/env python

"""Train a deep learning model according to a configuration file.

TODO enable continuing with training
TODO add seed
"""

import sys
import argparse
import yaml

from keras_transfer_learning import config
from keras_transfer_learning import backbone_configs
from keras_transfer_learning import head_configs
from keras_transfer_learning import data_configs
from keras_transfer_learning import train


def _parse_backbone_config(conf) -> config.BackboneConfig:
    if conf['name'] == 'unet':
        return backbone_configs.UnetBackboneConfig(conf['args'], conf['weights'])
    raise NotImplementedError(
        'The backbone {} is not implemented.'.format(conf['name']))


def _parse_head_config(conf) -> config.HeadConfig:
    if conf['name'] == 'fgbg-segm':
        return head_configs.FgBgSegmHeadConfig(conf['args'], conf['prepare_model_args'])
    raise NotImplementedError(
        'The head {} is not implemented.'.format(conf['name']))


def _parse_training_config(conf) -> config.TrainingConfig:
    return config.TrainingConfig(conf['batch_size'], conf['callbacks'])


def _parse_data_config(conf) -> config.DataConfig:
    # Define the normalizer
    if conf['normalizer'] == 'uint8-range':
        def normalizer(data):
            return data / 255
    else:
        raise NotImplementedError(
            'The normalizer {} is not implemented.'.format(conf['normalizer']))

    # Define the data config
    if conf['name'] == 'stardist-dsb2018':
        return data_configs.StarDistDSB2018DataConfig(
            train_val_split=conf['train_val_split'],
            training_size=conf['training_size'],
            normalizer=normalizer)
    raise NotImplementedError(
        'The data {} is not implemented.'.format(conf['name']))


def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('configfile', help='Config file',
                        type=argparse.FileType('r'))
    parser.add_argument(
        '-e', '--epochs', help='Number of epochs', type=int, required=True)
    # TODO add other arguments to control training
    # Especially continue training
    args = parser.parse_args(arguments)

    # Load the config yaml
    conf = yaml.load(args.configfile)
    print(conf)

    # Create the config objects
    conf_backbone = _parse_backbone_config(conf['backbone'])
    conf_head = _parse_head_config(conf['head'])
    conf_training = _parse_training_config(conf['training'])
    conf_data = _parse_data_config(conf['data'])
    conf_all = config.Config(conf['name'], conf['input_shape'],
                  conf_backbone, conf_head, conf_training, conf_data)

    # Run the training
    train.train(conf_all, args.epochs)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
