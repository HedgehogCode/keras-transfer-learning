#!/usr/bin/env python

"""Train a deep learning model according to a configuration file.
"""

import sys
import argparse
from yaml import safe_load as yaml_load

from keras_transfer_learning.config import config
from keras_transfer_learning.config import backbone_configs
from keras_transfer_learning.config import head_configs
from keras_transfer_learning.config import training_configs
from keras_transfer_learning.config import data_configs
from keras_transfer_learning import train


def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-b', '--backbone', help='Backbone config file',
                        type=argparse.FileType('r'), required=True)
    parser.add_argument('--head', help='Head config file',
                        type=argparse.FileType('r'), required=True)
    parser.add_argument('-t', '--training', help='Training config file',
                        type=argparse.FileType('r'), required=True)
    parser.add_argument('-d', '--data', help='Data config file',
                        type=argparse.FileType('r'), required=True)
    parser.add_argument('-n', '--name', help='Name of the model',
                        type=str, required=True)
    parser.add_argument('-i', '--input_shape', help='Input shape of the model (yaml)',
                        type=str, required=True)
    parser.add_argument('-e', '--epochs', help='Number of epochs',
                        type=int, required=True)
    # TODO add other arguments to control training
    # Especially continue training
    args = parser.parse_args(arguments)

    # Create the config objects
    conf_backbone = backbone_configs.get_config(yaml_load(args.backbone))
    conf_head = head_configs.get_config(yaml_load(args.head))
    conf_training = training_configs.get_config(yaml_load(args.training))
    conf_data = data_configs.get_config(yaml_load(args.data))
    conf = config.Config(args.name, yaml_load(args.input_shape),
                         conf_backbone, conf_head, conf_training, conf_data)

    # Run the training
    train.train(conf, epochs=args.epochs)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
