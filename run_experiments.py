#!/usr/bin/env python

"""Run all experiments
"""

import os
import sys
import argparse
from yaml import safe_load as yaml_load

import pandas as pd

from keras_transfer_learning import train, evaluate

CONFIG_FILES = {
    'backbone': {
        'resnet_unet': ['backbones', 'resnet-unet.yaml'],
        'unet_csbdeep': ['backbones', 'unet-csbdeep.yaml'],
        'unet_very_small': ['backbones', 'unet-very-small.yaml']
    },
    'data': {
        'cityscapes': ['data', 'cityscapes.yaml'],
        'dsb2018': ['data', 'dsb2018.yaml'],
        'granulocyte': ['data', 'granulocyte.yaml'],
        'hl60_high_noise': ['data', 'hl60_high-noise.yaml'],
        'hl60_low_noise': ['data', 'hl60_low-noise.yaml']
    },
    'head': {
        'fgbg_segm_weighted': ['heads', 'fgbg-segm-weighted.yaml'],
        'fgbg_segm': ['heads', 'fgbg-segm.yaml'],
        'segm_cityscapes': ['heads', 'segm_cityscapes.yaml'],
        'stardist': ['heads', 'stardist.yaml']
    },
    'training': {
        'default': ['training', 'bs-8_early-stopping_reduce-lr.yaml']
    }
}


def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-d', '--dry-run', action='store_true')
    args = parser.parse_args(arguments)
    dry_run = args.dry_run
    # TODO add arguments to run only a subset of the experiments


    configs = _get_configs()

    # Experiment 1: hl60 low and high noise
    _run_experiment_hl_60_low_high_noise(configs, dry_run)

    # TODO


def _run_experiment_hl_60_low_high_noise(configs, dry_run):
    max_epochs = 1000
    input_shape = [None, None, 1]
    conf_backbone = configs.backbone.unet_csbdeep
    conf_head = configs.head.stardist
    conf_training = configs.training.default
    conf_data_low_noise = configs.data.hl60_low_noise
    conf_data_high_noise = configs.data.hl60_high_noise

    # Step 1:
    # - Random init
    # - Low noise
    # - All data
    name ='E0_R_unet-stardist_hl60-low_F'
    _train_model({
        'name': name,
        'input_shape': input_shape,
        'backbone': conf_backbone,
        'head': conf_head,
        'training': conf_training,
        'data': conf_data_low_noise
    }, max_epochs, dry_run)
    _evaluate_model(name, dry_run)

    # Step 2:
    # - Random init
    # - High noise
    # - All data
    name ='E0_R_unet-stardist_hl60-high_F'
    _train_model({
        'name': name,
        'input_shape': input_shape,
        'backbone': conf_backbone,
        'head': conf_head,
        'training': conf_training,
        'data': conf_data_high_noise
    }, max_epochs, dry_run)
    _evaluate_model(name, dry_run)

    # Step 3:
    # - Random init
    # - High noise
    # - Parts of the data
    conf_data = conf_data_high_noise.copy()
    for num_train in [200, 50, 10, 5, 2]:
        conf_data['num_train'] = num_train
        name = 'E0_R_unet-stardist_hl60-high_{}'.format(num_train)
        _train_model({
            'name': name,
            'input_shape': input_shape,
            'backbone': conf_backbone,
            'head': conf_head,
            'training': conf_training,
            'data': conf_data
        }, max_epochs, dry_run)
        _evaluate_model(name, dry_run)

    # Step 4:
    # - Low-noise init
    # - High noise
    # - Parts of the data
    conf_backbone_pretrained = conf_backbone.copy()
    conf_backbone_pretrained['weights'] = os.path.join(
        'models', 'E0_R_unet-stardist_hl60-low_F', 'weights_final.h5')
    conf_data = conf_data_high_noise.copy()
    for num_train in [200, 50, 10, 5, 2]:
        conf_data['num_train'] = num_train
        name = 'E0_R_unet-stardist_hl60-high_{}'.format(num_train)
        _train_model({
            'name': name,
            'input_shape': input_shape,
            'backbone': conf_backbone_pretrained,
            'head': conf_head,
            'training': conf_training,
            'data': conf_data
        }, max_epochs, dry_run)
        _evaluate_model(name, dry_run)


def _train_model(conf, epochs, dry_run):
    if dry_run:
        print('Training model {} for {} epochs...'.format(
            conf['name'], epochs))
        print(conf)
        return

    if not os.path.isdir(os.path.join('models', conf['name'])):
        train.train(conf, epochs=epochs)
    else:
        print('Model {} already present.'.format(conf['name']))


def _evaluate_model(name, dry_run):
    if dry_run:
        print('Evaluating model {}...'.format(name))
        return

    with open(os.path.join('models', name, 'config.yaml'), 'r') as f:
        conf = yaml_load(f)

    results = {}
    epoch = 1
    while True:
        res = evaluate.evaluate(conf, epoch=epoch)

        if results == {}:
            # Create results dict
            results['epoch'] = [epoch]
            for k, v in res.items():
                results[k] = [v]
        else:
            # Update results dict
            results['epoch'].append(epoch)
            for k, v in res.items():
                results[k].append(v)

        epoch += 1

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join('models', name, 'results.csv'))


def _get_configs():
    configs = {}
    for type_key, type_val in CONFIG_FILES.items():
        type_configs = {}
        for key, val in type_val.items():
            with open(os.path.join('configs', *val), 'r') as f:
                type_configs[key] = yaml_load(f)
        configs[type_key] = argparse.Namespace(**type_configs)
    return argparse.Namespace(**configs)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
