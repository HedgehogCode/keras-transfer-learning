#!/usr/bin/env python

"""Evaluate a deep learning model according to a configuration file.
"""

import sys
import argparse
from yaml import unsafe_load as yaml_load

from keras_transfer_learning import evaluate


def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('configfile', help='Config file',
                        type=argparse.FileType('r'))
    # TODO add other arguments
    args = parser.parse_args(arguments)

    # Load the config yaml
    conf = yaml_load(args.configfile)

    # Run the evaluation
    evaluate.evaluate(conf)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
