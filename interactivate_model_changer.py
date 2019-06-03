#!/usr/bin/env python

"""Interactively rename models
"""

import os
import sys
import argparse
import glob
import shutil

import yaml
from yaml import unsafe_load as yaml_load

def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-f', '--filter', type=str, default='*')
    args = parser.parse_args(arguments)
    filter_glob = args.filter

    model_dirs = sorted(glob.glob(os.path.join('.', 'models', filter_glob)))

    for m in model_dirs:
        process_model(m)


def process_model(model_dir):
    with open(os.path.join(model_dir, 'config.yaml'), 'r') as f:
        conf = yaml_load(f)

    model_name = conf['name']

    print('Model name: "{}"'.format(model_name))

    while (True):
        inp = input('> ')

        if inp == 'n':
            return
        elif inp == 'r':
            # Rename model
            new_model_name = input("New model name: ")
            if new_model_name == model_name:
                print('The new model name is the same as the old model name.')
                continue

            new_model_dir = os.path.join('.', 'models', new_model_name)
            if os.path.exists(new_model_dir):
                print('A model with the name {} already exists. Aborting.'.format(new_model_name))
                continue

            # Change the config yaml
            conf['name'] = new_model_name
            with open(os.path.join(model_dir, 'config.yaml'), 'w') as f:
                yaml.dump(conf, f)

            # Move the model dir
            os.rename(model_dir, new_model_dir)
            model_dir = new_model_dir

            # Move log dir
            new_log_dir = os.path.join('.', 'logs', new_model_name)
            os.rename(os.path.join('.', 'logs', model_name), new_log_dir)

            model_name = new_model_name

            print('Renamed model. New name: "{}"'.format(model_name))

        elif inp == 'd':
            # Delete model
            confirm = input("Delete model? (yes/no): ")
            if confirm == 'yes':
                shutil.rmtree(model_dir)
                shutil.rmtree(os.path.join('.', 'logs', model_name))
                print('Model deleted.')

        elif inp == 'c':
            # Print config
            print(yaml.dump(conf))
        else:
            print('Unsupported command. Supported commands are "n", "r" and "c"')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
