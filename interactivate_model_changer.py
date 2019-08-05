#!/usr/bin/env python

"""Interactively rename models
"""

import os
import sys
import argparse
import shutil
import pandas as pd
import re

import yaml
from yaml import unsafe_load as yaml_load

from keras_transfer_learning import utils

MODEL_NAME_MAP = {}


def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-f', '--filter', type=str, default='.*')
    parser.add_argument('--auto-rename', action='store_true')
    parser.add_argument('--add-eval', action='store_true')
    parser.add_argument('--remove-results', action='store_true')
    parser.add_argument('--results-set-index', action='store_true')
    parser.add_argument('--auto-rename-weights', action='store_true')
    parser.add_argument('--auto-remove', action='store_true')
    args = parser.parse_args(arguments)
    filter_re = re.compile(args.filter)

    model_dirs = utils.utils.list_model_dirs()
    model_dirs = [d for d in model_dirs if filter_re.match(d)]

    for m in model_dirs:
        if args.auto_rename:
            auto_rename(m)
        elif args.add_eval:
            add_eval(m)
        elif args.remove_results:
            remove_results(m)
        elif args.results_set_index:
            results_set_index(m)
        elif args.auto_rename_weights:
            auto_rename_weights(m)
        elif args.auto_remove:
            auto_remove(m)
        else:
            process_model(m)

    print(MODEL_NAME_MAP)
    with open('model_map.yaml', 'w') as f:
        yaml.dump(MODEL_NAME_MAP, f)

def process_model(model_dir):
    with open(os.path.join(model_dir, 'config.yaml'), 'r') as f:
        conf = yaml_load(f)

    model_name = conf['name']

    print('Model name: "{}"'.format(model_name))
    try:
        utils.utils.get_last_weights(os.path.join('models', model_name))
    except:
        print("Couldn't find weights file")


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
                print('A model with the name {} already exists. Aborting.'.format(
                    new_model_name))
                continue

            # Change the config yaml
            conf['name'] = new_model_name
            with open(os.path.join(model_dir, 'config.yaml'), 'w') as f:
                yaml.dump(conf, f)

            # Move the model dir
            os.makedirs(new_model_dir[:new_model_dir.rfind(os.path.sep)], exist_ok=True)
            os.rename(model_dir, new_model_dir)
            model_dir = new_model_dir

            # Move log dir
            new_log_dir = os.path.join('.', 'logs', new_model_name)
            os.makedirs(new_log_dir[:new_log_dir.rfind(os.path.sep)], exist_ok=True)
            os.rename(os.path.join('.', 'logs', model_name), new_log_dir)

            model_name = new_model_name

            print('Renamed model. New name: "{}"'.format(model_name))

        elif inp == 'ra':
            # Rename automatically
            model_prefix, _, num_train = model_name.rpartition('_')
            if num_train == 'F':
                new_model_name = model_prefix + '_F'
            else:
                new_model_name = model_prefix + \
                    '_{:03d}'.format(int(num_train))

            if new_model_name == model_name:
                print('The new model name is the same as the old model name.')
                continue

            new_model_dir = os.path.join('.', 'models', new_model_name)
            if os.path.exists(new_model_dir):
                print('A model with the name {} already exists. Aborting.'.format(
                    new_model_name))
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


def auto_remove(model_dir):
    with open(os.path.join(model_dir, 'config.yaml'), 'r') as f:
        conf = yaml_load(f)

    model_name = conf['name']

    try:
        utils.utils.get_last_weights(model_dir)
        print(f"Found weights for model {model_name}")
    except ValueError:
        print(f"Couldn't find weights file for model {model_name}")
        shutil.rmtree(model_dir, ignore_errors=True)
        shutil.rmtree(os.path.join('.', 'logs', model_name), ignore_errors=True)
        print('Model deleted.')


def auto_rename(model_dir):
    with open(os.path.join(model_dir, 'config.yaml'), 'r') as f:
        conf = yaml_load(f)

    model_name = conf['name']

    print('Model name: "{}"'.format(model_name))

    # Rename model
    exp_name, _, model_suffix = model_name.partition('_')
    try:
        model_prefix = {
            'E1': 'A00',
            'E2': 'B00', 'E2a': 'B01', 'E2b': 'B02',
            'E3': 'C00', 'E3a': 'C01', 'E3b': 'C02',
            'E4': 'D00',
            'E5': 'E00',
            'E6': 'F00',
            'E7': 'G00',
        }[exp_name]
    except KeyError:
        return
    new_model_name = model_prefix + '_' + model_suffix

    if new_model_name == model_name:
        print('The new model name is the same as the old model name.')
        return

    new_model_dir = os.path.join('.', 'models', new_model_name)
    if os.path.exists(new_model_dir):
        print('A model with the name {} already exists. Aborting.'.format(
            new_model_name))
        return
    print(new_model_name)

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


def auto_rename_weights(model_dir):
    with open(os.path.join(model_dir, 'config.yaml'), 'r') as f:
        conf = yaml_load(f)

    model_name = conf['name']
    print('Model name: "{}"'.format(model_name))

    # Get the weights name
    weigths_name = conf['backbone']['weights']
    if weigths_name is None:
        print('No weigths')
        return

    if os.path.isfile(weigths_name):
        print('Weights file exists. No need to rename')
        return

    # Get the model name of the weigths
    weigths_model_name = weigths_name.split('/')[1]

    # Rename model
    exp_name, _, model_suffix = weigths_model_name.partition('_')
    try:
        model_prefix = {
            'E1': 'A00',
            'E2': 'B00', 'E2a': 'B01', 'E2b': 'B02',
            'E3': 'C00', 'E3a': 'C01', 'E3b': 'C02',
            'E4': 'D00',
            'E5': 'E00',
            'E6': 'F00',
            'E7': 'G00',
        }[exp_name]
    except KeyError:
        return
    new_weigths_model_name = model_prefix + '_' + model_suffix

    # Check if the weights model changed
    if new_weigths_model_name == weigths_model_name:
        print('ERROR: The new model name is the same as the old one ' +
              'but the file does not exist: {}'.format(new_weigths_model_name))
        new_weigths_model_name = input('New weigths model name: ')
        return

    # Get the new weigths file
    new_weights_file = os.path.join('models',
                                    new_weigths_model_name,
                                    weigths_name.split('/')[-1])
    if not os.path.isfile(new_weights_file):
        print('FATAL ERROR: The new weigths file does not exist: {}'.format(
            new_weights_file))
        exit()

    # Rename the old config yaml
    shutil.move(os.path.join(model_dir, 'config.yaml'),
                os.path.join(model_dir, 'config.yaml.bak2'))

    # Set to the new weights file
    conf['backbone']['weights'] = new_weights_file

    # Write the new config yaml
    with open(os.path.join(model_dir, 'config.yaml'), 'w') as f:
        yaml.dump(conf, f)

    print('Changed weights file to {}.'.format(new_weights_file))


def add_eval(model_dir):
    config_file = os.path.join(model_dir, 'config.yaml')
    with open(config_file, 'r') as f:
        conf = yaml_load(f)

    model_name = conf['name']

    print('Model name: "{}"'.format(model_name))

    if conf['head']['name'] in ['fgbg-segm', 'fgbg-segm-weighted', 'stardist']:
        # Load the instance segmentation evaluation
        with open(os.path.join('configs', 'eval', 'instance_segm.yaml'), 'r') as f:
            evaluation_conf = yaml_load(f)
    elif conf['head']['name'] in ['segm']:
        # Load the sematic segmentation evaluation
        with open(os.path.join('configs', 'eval', 'semantic_segm.yaml'), 'r') as f:
            evaluation_conf = yaml_load(f)
    else:
        print('WARN: Could not find evaluation for model.')
        return

    # Move the old config file
    shutil.move(config_file, config_file + '.bak')

    conf['evaluation'] = evaluation_conf

    # Write the new evaluation file
    with open(config_file, 'w') as f:
        yaml.dump(conf, f)

    print('Added evaluation to model')


def remove_results(model_dir):
    print('Model dir: "{}"'.format(model_dir))
    results_file = os.path.join(model_dir, 'results.csv')
    if os.path.isfile(results_file):
        shutil.move(results_file, results_file + '.bak')
        print('Moved the results file')
    print('No results file')


def results_set_index(model_dir):
    print('Model dir: "{}"'.format(model_dir))
    results_file = os.path.join(model_dir, 'results.csv')
    if os.path.isfile(results_file):
        results_df = pd.read_csv(results_file)
        results_df = results_df.drop('Unnamed: 0', axis=1)
        results_df = results_df.set_index('epoch')

        shutil.move(results_file, results_file + '.bak2')
        results_df.to_csv(results_file)
        print('Moved the results file and saved new')

    print('No results file')


def _rename_model(model_name, new_model_name):
    model_dir_old = os.path.join('models', model_name)
    model_dir_new = os.path.join('models', new_model_name)

    log_dir_old = os.path.join('logs', model_name)
    log_dir_new = os.path.join('logs', new_model_name)

    if os.path.isdir(model_dir_new):
        raise ValueError('The target model "{}" already exists.'.format(model_dir_new))
    if os.path.isdir(log_dir_new):
        raise ValueError('The target log directory "{}" already exists.'.format(log_dir_new))




if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
