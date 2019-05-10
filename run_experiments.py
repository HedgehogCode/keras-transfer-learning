#!/usr/bin/env python

"""Run all experiments 
"""

import sys
import argparse
from yaml import safe_load as yaml_load

from keras_transfer_learning import train

CONFIG_FILES = {
    'backbone': [
        ('resnet_unet', ['backbones', 'resnet-unet.yaml'])
    ]
    # TODO
}
CONFIGS = None


def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # TODO add arguments to run only a subset of the experiments

    # TODO

    # Run the training
    train.train(conf, epochs=args.epochs)


def _init_configs():
    if CONFIGS is None:
        # Backbones
        CONFIGS = None
        # TODO implement


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
