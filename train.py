#!/usr/bin/env python

"""Train a deep learning model according to a configuration file.
"""

from __future__ import print_function
import sys
import argparse
import yaml


def parse_args(arguments):
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('configfile', help="Config file",
                        type=argparse.FileType('r'))
    # TODO add other arguments to control training
    # Especially continue training

    return parser.parse_args(arguments)


def main(args):
    config = yaml.load(args.configfile)
    print(config)

    # TODO create the network and train it


if __name__ == '__main__':
    sys.exit(main(parse_args(sys.argv[1:])))
