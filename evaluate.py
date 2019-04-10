#!/usr/bin/env python

"""Evaluate a deep learning model according to a configuration file.
"""

import sys
import argparse
from yaml import safe_load as yaml_load

from keras_transfer_learning.config import config
from keras_transfer_learning.config import backbone_configs
from keras_transfer_learning.config import head_configs
from keras_transfer_learning.config import training_configs
from keras_transfer_learning.config import data_configs
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

    # Create the config objects
    conf_backbone = backbone_configs.get_config(conf['backbone'])
    conf_head = head_configs.get_config(conf['head'])
    conf_training = training_configs.get_config(conf['training'])
    conf_data = data_configs.get_config(conf['data'])
    conf_all = config.Config(conf['name'], conf['input_shape'],
                             conf_backbone, conf_head, conf_training, conf_data)

    # Run the evaluation
    evaluate.evaluate(conf_all)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
