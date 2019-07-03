#!/usr/bin/env python

"""Interactively rename models
"""

import os
import sys
import argparse
import glob
import shutil
import pandas as pd

import yaml
from yaml import unsafe_load as yaml_load


def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-f', '--filter', type=str, default='*')
    parser.add_argument('--auto-rename', action='store_true')
    parser.add_argument('--add-eval', action='store_true')
    parser.add_argument('--remove-results', action='store_true')
    parser.add_argument('--results-set-index', action='store_true')
    args = parser.parse_args(arguments)
    filter_glob = args.filter

    model_dirs = sorted(glob.glob(os.path.join('.', 'models', filter_glob)))

    for m in model_dirs:
        if args.auto_rename:
            auto_rename(m)
        elif args.add_eval:
            add_eval(m)
        elif args.remove_results:
            remove_results(m)
        elif args.results_set_index:
            results_set_index(m)
        else:
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


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
