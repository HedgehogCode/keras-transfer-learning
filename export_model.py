#!/usr/bin/env python

"""A python script which exports the
"""

import sys
import argparse
from yaml import unsafe_load as yaml_load

from keras_transfer_learning import model

def main(args):
    conf = yaml_load(args.configfile)
    m = model.Model(conf, load_weights='last', for_exporting=True)
    m.model.summary()
    m.export_to(args.outfile.name)


def parse_args(arguments):
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('configfile', help="Config file",
                        type=argparse.FileType('r'))
    parser.add_argument('outfile', help="Output file",
                        type=argparse.FileType('w'))

    return parser.parse_args(arguments)


if __name__ == '__main__':
    sys.exit(main(parse_args(sys.argv[1:])))
