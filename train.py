#!/usr/bin/env python

"""Train a deep learning model according to a configuration file.
"""

import sys
import argparse
from yaml import safe_load as yaml_load

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

    # Create the config dict
    conf = {
        'name': args.name,
        'input_shape': yaml_load(args.input_shape),
        'backbone': yaml_load(args.backbone),
        'head': yaml_load(args.head),
        'training': yaml_load(args.training),
        'data': yaml_load(args.data)
    }

    # Run the training
    train.train(conf, epochs=args.epochs)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
