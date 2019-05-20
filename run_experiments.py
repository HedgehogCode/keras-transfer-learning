#!/usr/bin/env python

"""Run all experiments
"""

import os
import sys
import argparse
from yaml import safe_load as yaml_load

import pandas as pd

import keras.backend as K

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
    parser.add_argument('--no-eval', action='store_true')
    args = parser.parse_args(arguments)
    dry_run = args.dry_run
    no_eval = args.no_eval
    # TODO add arguments to run only a subset of the experiments

    configs = _get_configs()

    # Experiment 1: hl60 low and high noise
    try:
        _run_experiment_hl_60_low_high_noise('E1', configs, dry_run, no_eval)
    except Exception as e:
        print("ERROR: Experiment E1 failed:", e)

    # Experiment 2: hl60 and granulocyte
    try:
        _run_experiment_hl_60_granulocyte('E2', configs, dry_run, no_eval)
    except Exception as e:
        print("ERROR: Experiment E2 failed:", e)

    # Experiment 3: hl60 and granulocyte
    try:
        _run_experiment_granulocyte_dsb2018('E3', configs, dry_run, no_eval)
    except Exception as e:
        print("ERROR: Experiment E3 failed:", e)

    # Experiment 4: hl60 and granulocyte
    try:
        _run_experiment_hl60_low_cityscapes('E4', configs, dry_run, no_eval)
    except Exception as e:
        print("ERROR: Experiment E4 failed:", e)

    # Experiment 5: hl60 and granulocyte
    try:
        _run_experiment_dsb2018_cityscapes('E5', configs, dry_run, no_eval)
    except Exception as e:
        print("ERROR: Experiment E5 failed:", e)

    # TODO


###################################################################################################
#   EXPERIMENT HL60 Low/High Noise
###################################################################################################

def _run_experiment_hl_60_low_high_noise(name, configs, dry_run, no_eval):
    conf_backbone = configs.backbone.unet_csbdeep
    conf_head = configs.head.stardist
    conf_training = configs.training.default
    conf_data_low_noise = configs.data.hl60_low_noise
    conf_data_high_noise = configs.data.hl60_high_noise

    _run_default_experiment(name, conf_training,
                            'unet', conf_backbone,
                            'stardist', conf_head,
                            'stardist', conf_head,
                            'hl60-low-noise', conf_data_low_noise,
                            'hl60-high-noise', conf_data_high_noise,
                            dry_run, no_eval)


###################################################################################################
#   EXPERIMENT HL60/Granulocyte
###################################################################################################

def _run_experiment_hl_60_granulocyte(name, configs, dry_run, no_eval):
    conf_backbone = configs.backbone.unet_csbdeep
    conf_head = configs.head.stardist
    conf_training = configs.training.default
    conf_data_low_noise = configs.data.hl60_low_noise
    conf_data_granulocyte = configs.data.granulocyte

    _run_default_experiment(name, conf_training,
                            'unet', conf_backbone,
                            'stardist', conf_head,
                            'stardist', conf_head,
                            'hl60-low-noise', conf_data_low_noise,
                            'granulocyte', conf_data_granulocyte,
                            dry_run, no_eval)


###################################################################################################
#   EXPERIMENT Granulocyte/DSB2018
###################################################################################################

def _run_experiment_granulocyte_dsb2018(name, configs, dry_run, no_eval):
    conf_backbone = configs.backbone.unet_csbdeep
    conf_head = configs.head.stardist
    conf_training = configs.training.default
    conf_data_granulocyte = configs.data.granulocyte
    conf_data_dsb2018 = configs.data.dsb2018

    _run_default_experiment(name, conf_training,
                            'unet', conf_backbone,
                            'stardist', conf_head,
                            'stardist', conf_head,
                            'granulocyte', conf_data_granulocyte,
                            'dsb2018', conf_data_dsb2018,
                            dry_run, no_eval)


###################################################################################################
#   EXPERIMENT HL60/Cityscapes
###################################################################################################

def _run_experiment_hl60_low_cityscapes(name, configs, dry_run, no_eval):
    conf_backbone = configs.backbone.resnet_unet
    conf_head_hl60 = configs.head.stardist
    conf_head_cityscapes = configs.head.segm_cityscapes
    conf_training = configs.training.default
    conf_data_hl60 = configs.data.hl60_low_noise
    conf_data_cityscapes = configs.data.cityscapes

    _run_default_experiment(name, conf_training,
                            'resnet-unet', conf_backbone,
                            'stardist', conf_head_hl60,
                            'segm', conf_head_cityscapes,
                            'hl60-low-noise', conf_data_hl60,
                            'cityscapes', conf_data_cityscapes,
                            dry_run, no_eval)


###################################################################################################
#   EXPERIMENT DSB2018/Cityscapes
###################################################################################################

def _run_experiment_dsb2018_cityscapes(name, configs, dry_run, no_eval):
    conf_backbone = configs.backbone.resnet_unet
    conf_head_dsb2018 = configs.head.stardist
    conf_head_cityscapes = configs.head.segm_cityscapes
    conf_training = configs.training.default
    conf_data_dsb2018 = configs.data.dsb2018
    conf_data_cityscapes = configs.data.cityscapes

    _run_default_experiment(name, conf_training,
                            'resnet-unet', conf_backbone,
                            'stardist', conf_head_dsb2018,
                            'segm', conf_head_cityscapes,
                            'dsb2018', conf_data_dsb2018,
                            'cityscapes', conf_data_cityscapes,
                            dry_run, no_eval)


###################################################################################################
#   Utils
###################################################################################################

def _run_default_experiment(name_experiment, conf_training,
                            name_backbone, conf_backbone,
                            name_head_1, conf_head_1,
                            name_head_2, conf_head_2,
                            name_data_1, conf_data_1,
                            name_data_2, conf_data_2,
                            dry_run, no_eval):
    max_epochs = 1000
    input_shape = [None, None, 1]
    # Step 1:
    # - Random init
    # - Head 1 + Data 1
    # - All data
    name_model_1 = _get_model_name(
        name_experiment, name_backbone, name_head_1, name_data_1, False, 'F')
    _train_model({
        'name': name_model_1,
        'input_shape': input_shape,
        'backbone': conf_backbone,
        'head': conf_head_1,
        'training': conf_training,
        'data': conf_data_1
    }, max_epochs, dry_run)
    if not no_eval:
        _evaluate_model(name_model_1, dry_run)

    # Step 2:
    # - Random init
    # - Head 2 + Data 2
    # - All data
    name_model_2 = _get_model_name(
        name_experiment, name_backbone, name_head_2, name_data_2, False, 'F')
    _train_model({
        'name': name_model_2,
        'input_shape': input_shape,
        'backbone': conf_backbone,
        'head': conf_head_2,
        'training': conf_training,
        'data': conf_data_2
    }, max_epochs, dry_run)
    if not no_eval:
        _evaluate_model(name_model_2, dry_run)

    # Step 3:
    # - Random init
    # - Head 1 + Data 1
    # - Parts of the data
    conf_data = conf_data_1.copy()
    for num_train in [200, 50, 10, 5, 2]:
        conf_data['num_train'] = num_train
        name = _get_model_name(
            name_experiment, name_backbone, name_head_1, name_data_1, False, num_train)
        _train_model({
            'name': name,
            'input_shape': input_shape,
            'backbone': conf_backbone,
            'head': conf_head_1,
            'training': conf_training,
            'data': conf_data
        }, max_epochs, dry_run)
        if not no_eval:
            _evaluate_model(name, dry_run)

    # Step 4:
    # - Random init
    # - Head 2 + Data 2
    # - Parts of the data
    conf_data = conf_data_2.copy()
    for num_train in [200, 50, 10, 5, 2]:
        conf_data['num_train'] = num_train
        name = _get_model_name(
            name_experiment, name_backbone, name_head_2, name_data_2, False, num_train)
        _train_model({
            'name': name,
            'input_shape': input_shape,
            'backbone': conf_backbone,
            'head': conf_head_2,
            'training': conf_training,
            'data': conf_data
        }, max_epochs, dry_run)
        if not no_eval:
            _evaluate_model(name, dry_run)

    # Step 5:
    # - Step 1 model init
    # - Head 2 + Data 2
    # - Parts of the data
    conf_backbone_pretrained = conf_backbone.copy()
    conf_backbone_pretrained['weights'] = os.path.join(
        'models', name_model_1, 'weights_final.h5')
    conf_data = conf_data_2.copy()
    for num_train in [200, 50, 10, 5, 2]:
        conf_data['num_train'] = num_train
        name = _get_model_name(
            name_experiment, name_backbone, name_head_2, name_data_2, True, num_train)
        _train_model({
            'name': name,
            'input_shape': input_shape,
            'backbone': conf_backbone_pretrained,
            'head': conf_head_2,
            'training': conf_training,
            'data': conf_data
        }, max_epochs, dry_run)
        if not no_eval:
            _evaluate_model(name, dry_run)

    # Step 6:
    # - Step 2 model init
    # - Head 1 + Data 1
    # - Parts of the data
    conf_backbone_pretrained = conf_backbone.copy()
    conf_backbone_pretrained['weights'] = os.path.join(
        'models', name_model_2, 'weights_final.h5')
    conf_data = conf_data_1.copy()
    for num_train in [200, 50, 10, 5, 2]:
        conf_data['num_train'] = num_train
        name = _get_model_name(
            name_experiment, name_backbone, name_head_1, name_data_1, True, num_train)
        _train_model({
            'name': name,
            'input_shape': input_shape,
            'backbone': conf_backbone_pretrained,
            'head': conf_head_1,
            'training': conf_training,
            'data': conf_data
        }, max_epochs, dry_run)
        if not no_eval:
            _evaluate_model(name, dry_run)


def _get_model_name(name_experiment, name_backbone, name_head, name_data, pretrained, num_train):
    return '{}_{}_{}_{}_{}_{}'.format(name_experiment, name_backbone, name_head, name_data,
                                      'P' if pretrained else 'R', num_train)


def _train_model(conf, epochs, dry_run):
    if dry_run:
        print('Training model {} for {} epochs...'.format(
            conf['name'], epochs))
        print(conf)
        return

    if not os.path.isdir(os.path.join('models', conf['name'])):
        try:
            K.clear_session()
            train.train(conf, epochs=epochs)
        except Exception as e:
            print('ERROR: Training of model {} failed: {}'.format(
                conf['name'], e))
    else:
        print('Model {} already present.'.format(conf['name']))


def _evaluate_model(name, dry_run):
    if dry_run:
        print('Evaluating model {}...'.format(name))
        return

    with open(os.path.join('models', name, 'config.yaml'), 'r') as f:
        conf = yaml_load(f)

    try:
        results = {}
        epoch = 1
        while True:
            try:
                K.clear_session()
                res = evaluate.evaluate(conf, epoch=epoch)
            except ValueError:
                # Last epoch
                break

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
    except Exception as e:
        print('ERROR: Evaluation of model {} failed: {}'.format(name, e))


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
