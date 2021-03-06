#!/usr/bin/env python

"""Train a deep learning model according to a configuration file.
"""

import sys
import argparse
from yaml import unsafe_load as yaml_load

from keras_transfer_learning import train


def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('configfile', help='Config file',
                        type=argparse.FileType('r'))
    parser.add_argument('-e', '--epochs', help='Number of epochs',
                        type=int, required=True)
    parser.add_argument('-i', '--initial_epoch', help='The initial epoch',
                        type=int, default=0)
    # TODO add other arguments to control training
    # Especially continue training
    args = parser.parse_args(arguments)

    # Load the config yaml
    conf = yaml_load(args.configfile)

    # Run the training
    train.train(conf, epochs=args.epochs, initial_epoch=args.initial_epoch)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
