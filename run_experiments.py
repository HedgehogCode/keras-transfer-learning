#!/usr/bin/env python

"""Run all experiments
"""

import os
import sys
import argparse
import traceback
from yaml import unsafe_load as yaml_load

import pandas as pd

from keras_transfer_learning import train, evaluate, utils

CONFIG_FILES = {
    'backbone': {
        'resnet_unet': ['backbones', 'resnet-unet.yaml'],
        'plain_unet': ['backbones', 'plain-unet.yaml'],
        'imagenet_resnet_unet': ['backbones', 'imagenet-resnet-unet.yaml'],
        'imagenet_resnet_unet_random': ['backbones', 'imagenet-resnet-unet-random.yaml'],
        'resnet_unet_big': ['backbones', 'resnet-unet-big.yaml'],
        'unet_csbdeep': ['backbones', 'unet-csbdeep.yaml'],
        'unet_very_small': ['backbones', 'unet-very-small.yaml']
    },
    'data': {
        'cityscapes': ['data', 'cityscapes.yaml'],
        'dsb2018': ['data', 'dsb2018.yaml'],
        'dsb2018_heavy_aug': ['data', 'dsb2018-heavy-aug.yaml'],
        'granulocyte': ['data', 'granulocyte.yaml'],
        'hl60_high_noise': ['data', 'hl60_high-noise.yaml'],
        'hl60_low_noise': ['data', 'hl60_low-noise.yaml'],
        'hl60_aug': ['data', 'hl60-aug.yaml'],
        'granulocyte_aug': ['data', 'granulocyte-aug.yaml'],
        'stardist_dsb2018': ['data', 'stardist-dsb2018.yaml'],
    },
    'head': {
        'fgbg_segm_weighted': ['heads', 'fgbg-segm-weighted.yaml'],
        'fgbg_segm': ['heads', 'fgbg-segm.yaml'],
        'segm_cityscapes': ['heads', 'segm_cityscapes.yaml'],
        'stardist': ['heads', 'stardist.yaml']
    },
    'training': {
        'default': ['training', 'bs-8_early-stopping_reduce-lr.yaml'],
        'small_bs': ['training', 'bs-2_early-stopping_reduce-lr.yaml'],
        'small_bs_less_cp': ['training', 'bs-2_early-stopping_reduce-lr_remove-checkpoints.yaml']
    },
    'evaluation': {
        'instance_segm': ['eval', 'instance_segm.yaml'],
        'semantic_segm': ['eval', 'semantic_segm.yaml']
    }
}


def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-d', '--dry-run', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--no-eval', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(arguments)

    configs = _get_configs()

    _run_random_init_models(configs, args)

    _run_frankenstein_models(configs, args)

    _run_pretrained_models(configs, args)


def _run_random_init_models(configs, args):
    num_train_options = ['F', 200, 50, 10, 5, 2]

    # ------------------------------------------------
    # HL60 Low Noise
    # ------------------------------------------------
    # Unet
    num_experiments = 15
    _train_eval_random_init_models('hl60low', configs.data.hl60_low_noise,
                                   'stardist', configs.head.stardist,
                                   'unet', configs.backbone.unet_csbdeep,
                                   configs.training.default,
                                   configs.evaluation.instance_segm,
                                   num_train_options,
                                   num_experiments,
                                   args)

    # ResNet Unet
    num_experiments = 5
    _train_eval_random_init_models('hl60low', configs.data.hl60_low_noise,
                                   'stardist', configs.head.stardist,
                                   'resnet-unet', configs.backbone.resnet_unet,
                                   configs.training.default,
                                   configs.evaluation.instance_segm,
                                   num_train_options,
                                   num_experiments,
                                   args)

    # ------------------------------------------------
    # HL60 High Noise
    # ------------------------------------------------
    num_experiments = 5
    _train_eval_random_init_models('hl60high', configs.data.hl60_high_noise,
                                   'stardist', configs.head.stardist,
                                   'unet', configs.backbone.unet_csbdeep,
                                   configs.training.default,
                                   configs.evaluation.instance_segm,
                                   num_train_options,
                                   num_experiments,
                                   args)

    # ------------------------------------------------
    # Granulocyte
    # ------------------------------------------------
    # Stardist - Unet
    num_experiments = 10
    _train_eval_random_init_models('granulocyte', configs.data.granulocyte,
                                   'stardist', configs.head.stardist,
                                   'unet', configs.backbone.unet_csbdeep,
                                   configs.training.default,
                                   configs.evaluation.instance_segm,
                                   num_train_options,
                                   num_experiments,
                                   args)

    # FGBG Weighted - ResNet Unet
    num_experiments = 3
    _train_eval_random_init_models('granulocyte', configs.data.granulocyte,
                                   'fgbg-weighted', configs.head.fgbg_segm_weighted,
                                   'resnet-unet', configs.backbone.resnet_unet,
                                   configs.training.default,
                                   configs.evaluation.instance_segm,
                                   num_train_options,
                                   num_experiments,
                                   args)

    # ------------------------------------------------
    # DSB2018
    # ------------------------------------------------
    # Stardist - Unet
    num_experiments = 10
    _train_eval_random_init_models('dsb2018', configs.data.dsb2018,
                                   'stardist', configs.head.stardist,
                                   'unet', configs.backbone.unet_csbdeep,
                                   configs.training.default,
                                   configs.evaluation.instance_segm,
                                   num_train_options,
                                   num_experiments,
                                   args)

    # Stardist - ResNet Unet
    num_experiments = 10
    _train_eval_random_init_models('dsb2018', configs.data.dsb2018,
                                   'stardist', configs.head.stardist,
                                   'resnet-unet', configs.backbone.resnet_unet,
                                   configs.training.default,
                                   configs.evaluation.instance_segm,
                                   num_train_options,
                                   num_experiments,
                                   args)

    # Stardist - Plain Unet
    num_experiments = 10
    _train_eval_random_init_models('dsb2018', configs.data.dsb2018,
                                   'stardist', configs.head.stardist,
                                   'plain-unet', configs.backbone.plain_unet,
                                   configs.training.default,
                                   configs.evaluation.instance_segm,
                                   ['F'],
                                   num_experiments,
                                   args)

    # FGBG Weighted - ResNet Unet
    num_experiments = 3
    _train_eval_random_init_models('dsb2018', configs.data.dsb2018,
                                   'fgbg-weighted', configs.head.fgbg_segm_weighted,
                                   'resnet-unet', configs.backbone.resnet_unet,
                                   configs.training.default,
                                   configs.evaluation.instance_segm,
                                   num_train_options,
                                   num_experiments,
                                   args)

    # StarDist - ResNet50 Unet
    num_experiments = 3
    _train_eval_random_init_models('dsb2018', configs.data.dsb2018,
                                   'stardist', configs.head.stardist,
                                   'imagenet-resnet-unet', configs.backbone.imagenet_resnet_unet_random,
                                   configs.training.small_bs_less_cp,
                                   configs.evaluation.instance_segm,
                                   num_train_options,
                                   num_experiments,
                                   args)

    # FGBG Weighted - ResNet Unet Big (Monster)
    # TODO fix nice data augmentation
    # num_experiments = 1
    # _train_eval_random_init_models('dsb2018', configs.data.dsb2018,
    #                                'fgbg-weighted', configs.head.fgbg_segm_weighted,
    #                                'resnet-unet-big', configs.backbone.resnet_unet_big,
    #                                configs.training.small_bs,
    #                                configs.evaluation.instance_segm,
    #                                ['F'],
    #                                num_experiments,
    #                                args)

    # ------------------------------------------------
    # STARDIST_DSB2018
    # ------------------------------------------------
    # Stardist - Unet
    num_experiments = 6
    _train_eval_random_init_models('stardist-dsb2018', configs.data.stardist_dsb2018,
                                   'stardist', configs.head.stardist,
                                   'unet', configs.backbone.unet_csbdeep,
                                   configs.training.default,
                                   configs.evaluation.instance_segm,
                                   ['F'],
                                   num_experiments,
                                   args)

    # Stardist - ResNet Unet
    num_experiments = 6
    _train_eval_random_init_models('stardist-dsb2018', configs.data.stardist_dsb2018,
                                   'stardist', configs.head.stardist,
                                   'resnet-unet', configs.backbone.resnet_unet,
                                   configs.training.default,
                                   configs.evaluation.instance_segm,
                                   ['F'],
                                   num_experiments,
                                   args)

    # Stardist - Plain Unet
    num_experiments = 6
    _train_eval_random_init_models('stardist-dsb2018', configs.data.stardist_dsb2018,
                                   'stardist', configs.head.stardist,
                                   'plain-unet', configs.backbone.plain_unet,
                                   configs.training.default,
                                   configs.evaluation.instance_segm,
                                   ['F'],
                                   num_experiments,
                                   args)

    # ------------------------------------------------
    # Cityscapes
    # ------------------------------------------------
    num_experiments = 5
    _train_eval_random_init_models('cityscapes', configs.data.cityscapes,
                                   'segm', configs.head.segm_cityscapes,
                                   'resnet-unet', configs.backbone.resnet_unet,
                                   configs.training.default,
                                   configs.evaluation.semantic_segm,
                                   num_train_options,
                                   num_experiments,
                                   args)


def _run_frankenstein_models(configs, args):
    # No data augmentation
    conf_datas = [
        ('hl60low', configs.data.hl60_low_noise),
        ('hl60high', configs.data.hl60_high_noise),
        ('granulocyte', configs.data.granulocyte),
        ('hl60low', configs.data.hl60_low_noise),
        ('hl60high', configs.data.hl60_high_noise),
        ('granulocyte', configs.data.granulocyte),
        ('hl60low', configs.data.hl60_low_noise),
        ('hl60high', configs.data.hl60_high_noise),
        ('granulocyte', configs.data.granulocyte),
        ('hl60low', configs.data.hl60_low_noise),
        ('hl60high', configs.data.hl60_high_noise),
        ('granulocyte', configs.data.granulocyte),
        ('hl60low', configs.data.hl60_low_noise),
        ('hl60high', configs.data.hl60_high_noise),
        ('granulocyte', configs.data.granulocyte)
    ]
    num_experiments = 5
    epochs_per_model = 15
    _train_frankenstein_models('hl60low-hl60high-granulocyte', conf_datas,
                               'stardist', configs.head.stardist,
                               'resnet-unet', configs.backbone.resnet_unet,
                               configs.training.default,
                               configs.evaluation.instance_segm,
                               epochs_per_model, num_experiments, args)

    # Data augmentation
    conf_datas = [
        ('hl60low', configs.data.hl60_aug),
        ('granulocyte', configs.data.granulocyte_aug),
        ('hl60low', configs.data.hl60_aug),
        ('granulocyte', configs.data.granulocyte_aug),
        ('hl60low', configs.data.hl60_aug),
        ('granulocyte', configs.data.granulocyte_aug),
        ('hl60low', configs.data.hl60_aug),
        ('granulocyte', configs.data.granulocyte_aug),
        ('hl60low', configs.data.hl60_aug),
        ('granulocyte', configs.data.granulocyte_aug),
        ('hl60low', configs.data.hl60_aug),
        ('granulocyte', configs.data.granulocyte_aug),
        ('hl60low', configs.data.hl60_aug),
        ('granulocyte', configs.data.granulocyte_aug),
        ('hl60low', configs.data.hl60_aug),
        ('granulocyte', configs.data.granulocyte_aug),
        ('hl60low', configs.data.hl60_aug),
        ('granulocyte', configs.data.granulocyte_aug),
        ('hl60low', configs.data.hl60_aug),
        ('granulocyte', configs.data.granulocyte_aug),
    ]
    num_experiments = 5
    epochs_per_model = 15
    _train_frankenstein_models('hl60low-granulocyte-aug', conf_datas,
                               'stardist', configs.head.stardist,
                               'resnet-unet', configs.backbone.resnet_unet,
                               configs.training.default,
                               configs.evaluation.instance_segm,
                               epochs_per_model, num_experiments, args)


def _run_pretrained_models(configs, args):
    num_train_options = [200, 50, 10, 5, 2]

    # ------------------------------------------------
    # HL60 Low Noise
    # ------------------------------------------------
    # HL60 High Noise
    model_names_pretrained = [
        'R/none/hl60low/stardist/unet/F/000',
        'R/none/hl60low/stardist/unet/F/001',
        'R/none/hl60low/stardist/unet/F/002',
        'R/none/hl60low/stardist/unet/F/003',
        'R/none/hl60low/stardist/unet/F/004',
    ]
    _train_eval_pretrained_models('hl60high', configs.data.hl60_high_noise,
                                  'stardist', configs.head.stardist,
                                  'unet', configs.backbone.unet_csbdeep,
                                  configs.training.default,
                                  configs.evaluation.instance_segm,
                                  num_train_options,
                                  model_names_pretrained,
                                  args)

    # Granulocyte
    model_names_pretrained = [
        'R/none/hl60low/stardist/unet/F/005',
        'R/none/hl60low/stardist/unet/F/006',
        'R/none/hl60low/stardist/unet/F/007',
        'R/none/hl60low/stardist/unet/F/008',
        'R/none/hl60low/stardist/unet/F/009',
        'R/none/hl60low/stardist/unet/F/005',
        'R/none/hl60low/stardist/unet/F/006',
        'R/none/hl60low/stardist/unet/F/007',
        'R/none/hl60low/stardist/unet/F/008',
        'R/none/hl60low/stardist/unet/F/009',
    ]
    _train_eval_pretrained_models('granulocyte', configs.data.granulocyte,
                                  'stardist', configs.head.stardist,
                                  'unet', configs.backbone.unet_csbdeep,
                                  configs.training.default,
                                  configs.evaluation.instance_segm,
                                  num_train_options,
                                  model_names_pretrained,
                                  args)

    # DSB2018
    model_names_pretrained = [
        'R/none/hl60low/stardist/unet/F/010',
        'R/none/hl60low/stardist/unet/F/011',
        'R/none/hl60low/stardist/unet/F/012',
        'R/none/hl60low/stardist/unet/F/013',
        'R/none/hl60low/stardist/unet/F/014',
        'R/none/hl60low/stardist/unet/F/010',
        'R/none/hl60low/stardist/unet/F/011',
        'R/none/hl60low/stardist/unet/F/012',
        'R/none/hl60low/stardist/unet/F/013',
        'R/none/hl60low/stardist/unet/F/014',
    ]
    _train_eval_pretrained_models('dsb2018', configs.data.dsb2018,
                                  'stardist', configs.head.stardist,
                                  'unet', configs.backbone.unet_csbdeep,
                                  configs.training.default,
                                  configs.evaluation.instance_segm,
                                  num_train_options,
                                  model_names_pretrained,
                                  args)

    # Cityscapes
    model_names_pretrained = [
        'R/none/hl60low/stardist/resnet-unet/F/000',
    ]
    _train_eval_pretrained_models('cityscapes', configs.data.cityscapes,
                                  'segm', configs.head.segm_cityscapes,
                                  'resnet-unet', configs.backbone.resnet_unet,
                                  configs.training.default,
                                  configs.evaluation.semantic_segm,
                                  num_train_options,
                                  model_names_pretrained,
                                  args)

    # ------------------------------------------------
    # HL60 High Noise
    # ------------------------------------------------
    # HL60 Low Noise
    model_names_pretrained = [
        'R/none/hl60high/stardist/unet/F/000',
        'R/none/hl60high/stardist/unet/F/001',
        'R/none/hl60high/stardist/unet/F/002',
        'R/none/hl60high/stardist/unet/F/003',
        'R/none/hl60high/stardist/unet/F/004',
    ]
    _train_eval_pretrained_models('hl60low', configs.data.hl60_low_noise,
                                  'stardist', configs.head.stardist,
                                  'unet', configs.backbone.unet_csbdeep,
                                  configs.training.default,
                                  configs.evaluation.instance_segm,
                                  num_train_options,
                                  model_names_pretrained,
                                  args)

    # Granulocyte
    model_names_pretrained = [
        'R/none/hl60high/stardist/unet/F/000',
        'R/none/hl60high/stardist/unet/F/001',
        'R/none/hl60high/stardist/unet/F/002',
        'R/none/hl60high/stardist/unet/F/003',
        'R/none/hl60high/stardist/unet/F/004',
    ]
    _train_eval_pretrained_models('granulocyte', configs.data.granulocyte,
                                  'stardist', configs.head.stardist,
                                  'unet', configs.backbone.unet_csbdeep,
                                  configs.training.default,
                                  configs.evaluation.instance_segm,
                                  num_train_options,
                                  model_names_pretrained,
                                  args)

    # DSB2018
    model_names_pretrained = [
        'R/none/hl60high/stardist/unet/F/000',
        'R/none/hl60high/stardist/unet/F/001',
        'R/none/hl60high/stardist/unet/F/002',
        'R/none/hl60high/stardist/unet/F/003',
        'R/none/hl60high/stardist/unet/F/004',
    ]
    _train_eval_pretrained_models('dsb2018', configs.data.dsb2018,
                                  'stardist', configs.head.stardist,
                                  'unet', configs.backbone.unet_csbdeep,
                                  configs.training.default,
                                  configs.evaluation.instance_segm,
                                  num_train_options,
                                  model_names_pretrained,
                                  args)

    # ------------------------------------------------
    # Granulocyte
    # ------------------------------------------------
    # HL60 Low Noise
    model_names_pretrained = [
        'R/none/granulocyte/stardist/unet/F/000',
        'R/none/granulocyte/stardist/unet/F/001',
        'R/none/granulocyte/stardist/unet/F/002',
        'R/none/granulocyte/stardist/unet/F/003',
        'R/none/granulocyte/stardist/unet/F/004',
        'R/none/granulocyte/stardist/unet/F/005',
        'R/none/granulocyte/stardist/unet/F/006',
        'R/none/granulocyte/stardist/unet/F/007',
        'R/none/granulocyte/stardist/unet/F/008',
        'R/none/granulocyte/stardist/unet/F/009',
    ]
    _train_eval_pretrained_models('hl60low', configs.data.hl60_low_noise,
                                  'stardist', configs.head.stardist,
                                  'unet', configs.backbone.unet_csbdeep,
                                  configs.training.default,
                                  configs.evaluation.instance_segm,
                                  num_train_options,
                                  model_names_pretrained,
                                  args)

    # HL60 Low Noise
    model_names_pretrained = [
        'R/none/granulocyte/stardist/unet/F/000',
        'R/none/granulocyte/stardist/unet/F/001',
        'R/none/granulocyte/stardist/unet/F/002',
        'R/none/granulocyte/stardist/unet/F/003',
        'R/none/granulocyte/stardist/unet/F/004',
        # 'R/none/granulocyte/stardist/unet/F/005',
        # 'R/none/granulocyte/stardist/unet/F/006',
        # 'R/none/granulocyte/stardist/unet/F/007',
        # 'R/none/granulocyte/stardist/unet/F/008',
        # 'R/none/granulocyte/stardist/unet/F/009',
    ]
    _train_eval_pretrained_models('hl60high', configs.data.hl60_high_noise,
                                  'stardist', configs.head.stardist,
                                  'unet', configs.backbone.unet_csbdeep,
                                  configs.training.default,
                                  configs.evaluation.instance_segm,
                                  num_train_options,
                                  model_names_pretrained,
                                  args)

    # DSB2018
    model_names_pretrained = [
        'R/none/granulocyte/stardist/unet/F/005',
        'R/none/granulocyte/stardist/unet/F/006',
        'R/none/granulocyte/stardist/unet/F/007',
        'R/none/granulocyte/stardist/unet/F/008',
        'R/none/granulocyte/stardist/unet/F/009',
        'R/none/granulocyte/stardist/unet/F/005',
        'R/none/granulocyte/stardist/unet/F/006',
        'R/none/granulocyte/stardist/unet/F/007',
        'R/none/granulocyte/stardist/unet/F/008',
        'R/none/granulocyte/stardist/unet/F/009',
    ]
    _train_eval_pretrained_models('dsb2018', configs.data.dsb2018,
                                  'stardist', configs.head.stardist,
                                  'unet', configs.backbone.unet_csbdeep,
                                  configs.training.default,
                                  configs.evaluation.instance_segm,
                                  num_train_options,
                                  model_names_pretrained,
                                  args)

    # DSB2018
    model_names_pretrained = [
        'R/none/granulocyte/fgbg-weighted/resnet-unet/F/000',
        'R/none/granulocyte/fgbg-weighted/resnet-unet/F/001',
        'R/none/granulocyte/fgbg-weighted/resnet-unet/F/002',
    ]
    _train_eval_pretrained_models('dsb2018', configs.data.dsb2018,
                                  'fgbg-weighted', configs.head.fgbg_segm_weighted,
                                  'resnet-unet', configs.backbone.resnet_unet,
                                  configs.training.default,
                                  configs.evaluation.instance_segm,
                                  num_train_options,
                                  model_names_pretrained,
                                  args)

    # ------------------------------------------------
    # DSB2018
    # ------------------------------------------------
    # Granulocyte - StarDist - Unet
    model_names_pretrained = [
        'R/none/dsb2018/stardist/unet/F/000',
        'R/none/dsb2018/stardist/unet/F/001',
        'R/none/dsb2018/stardist/unet/F/002',
        'R/none/dsb2018/stardist/unet/F/003',
        'R/none/dsb2018/stardist/unet/F/004',
        'R/none/dsb2018/stardist/unet/F/005',
        'R/none/dsb2018/stardist/unet/F/006',
        'R/none/dsb2018/stardist/unet/F/007',
        'R/none/dsb2018/stardist/unet/F/008',
        'R/none/dsb2018/stardist/unet/F/009',
    ]
    _train_eval_pretrained_models('granulocyte', configs.data.granulocyte,
                                  'stardist', configs.head.stardist,
                                  'unet', configs.backbone.unet_csbdeep,
                                  configs.training.default,
                                  configs.evaluation.instance_segm,
                                  num_train_options,
                                  model_names_pretrained,
                                  args)

    # HL60 Low Noise - StarDist - Unet
    model_names_pretrained = [
        'R/none/dsb2018/stardist/unet/F/005',
        'R/none/dsb2018/stardist/unet/F/006',
        'R/none/dsb2018/stardist/unet/F/007',
        'R/none/dsb2018/stardist/unet/F/008',
        'R/none/dsb2018/stardist/unet/F/009',
        'R/none/dsb2018/stardist/unet/F/005',
        'R/none/dsb2018/stardist/unet/F/006',
        'R/none/dsb2018/stardist/unet/F/007',
        'R/none/dsb2018/stardist/unet/F/008',
        'R/none/dsb2018/stardist/unet/F/009',
    ]
    _train_eval_pretrained_models('hl60low', configs.data.hl60_low_noise,
                                  'stardist', configs.head.stardist,
                                  'unet', configs.backbone.unet_csbdeep,
                                  configs.training.default,
                                  configs.evaluation.instance_segm,
                                  num_train_options,
                                  model_names_pretrained,
                                  args)

    # HL60 High Noise - StarDist - Unet
    model_names_pretrained = [
        'R/none/dsb2018/stardist/unet/F/000',
        'R/none/dsb2018/stardist/unet/F/001',
        'R/none/dsb2018/stardist/unet/F/002',
        'R/none/dsb2018/stardist/unet/F/003',
        'R/none/dsb2018/stardist/unet/F/004',
        # 'R/none/dsb2018/stardist/unet/F/005',
        # 'R/none/dsb2018/stardist/unet/F/006',
        # 'R/none/dsb2018/stardist/unet/F/007',
        # 'R/none/dsb2018/stardist/unet/F/008',
        # 'R/none/dsb2018/stardist/unet/F/009',
    ]
    _train_eval_pretrained_models('hl60high', configs.data.hl60_high_noise,
                                  'stardist', configs.head.stardist,
                                  'unet', configs.backbone.unet_csbdeep,
                                  configs.training.default,
                                  configs.evaluation.instance_segm,
                                  num_train_options,
                                  model_names_pretrained,
                                  args)

    # Granulocyte - fgbg-weighted - resnet-unet
    model_names_pretrained = [
        'R/none/dsb2018/fgbg-weighted/resnet-unet/F/000',
        'R/none/dsb2018/fgbg-weighted/resnet-unet/F/001',
        'R/none/dsb2018/fgbg-weighted/resnet-unet/F/002',
    ]
    _train_eval_pretrained_models('granulocyte', configs.data.granulocyte,
                                  'fgbg-weighted', configs.head.fgbg_segm_weighted,
                                  'resnet-unet', configs.backbone.resnet_unet,
                                  configs.training.default,
                                  configs.evaluation.instance_segm,
                                  num_train_options,
                                  model_names_pretrained,
                                  args)

    # Cityscapes - segm - resnet-unet
    model_names_pretrained = [
        'R/none/dsb2018/stardist/resnet-unet/F/000',
    ]
    _train_eval_pretrained_models('cityscapes', configs.data.cityscapes,
                                  'segm', configs.head.segm_cityscapes,
                                  'resnet-unet', configs.backbone.resnet_unet,
                                  configs.training.default,
                                  configs.evaluation.semantic_segm,
                                  num_train_options,
                                  model_names_pretrained,
                                  args)

    # ------------------------------------------------
    # Cityscapes
    # ------------------------------------------------
    # HL60 Low Noise - stardist - resnet-unet
    model_names_pretrained = [
        'R/none/cityscapes/segm/resnet-unet/F/000',
        'R/none/cityscapes/segm/resnet-unet/F/001',
        'R/none/cityscapes/segm/resnet-unet/F/002',
        'R/none/cityscapes/segm/resnet-unet/F/003',
        'R/none/cityscapes/segm/resnet-unet/F/004',
    ]
    _train_eval_pretrained_models('hl60low', configs.data.hl60_low_noise,
                                  'stardist', configs.head.stardist,
                                  'resnet-unet', configs.backbone.resnet_unet,
                                  configs.training.default,
                                  configs.evaluation.instance_segm,
                                  num_train_options,
                                  model_names_pretrained,
                                  args)

    # DSB2018 - stardist - resnet-unet
    model_names_pretrained = [
        'R/none/cityscapes/segm/resnet-unet/F/001',
        'R/none/cityscapes/segm/resnet-unet/F/001',
        'R/none/cityscapes/segm/resnet-unet/F/002',
        'R/none/cityscapes/segm/resnet-unet/F/003',
        'R/none/cityscapes/segm/resnet-unet/F/004',
    ]
    _train_eval_pretrained_models('dsb2018', configs.data.dsb2018,
                                  'stardist', configs.head.stardist,
                                  'resnet-unet', configs.backbone.resnet_unet,
                                  configs.training.default,
                                  configs.evaluation.instance_segm,
                                  num_train_options,
                                  model_names_pretrained,
                                  args)

    # ------------------------------------------------
    # Frankenstein - No aug
    # ------------------------------------------------
    model_names_pretrained = [
        'T/none/hl60low-hl60high-granulocyte/stardist/resnet-unet/F/000/015_granulocyte',
        'T/none/hl60low-hl60high-granulocyte/stardist/resnet-unet/F/001/015_granulocyte',
        'T/none/hl60low-hl60high-granulocyte/stardist/resnet-unet/F/002/015_granulocyte',
        'T/none/hl60low-hl60high-granulocyte/stardist/resnet-unet/F/003/015_granulocyte',
        'T/none/hl60low-hl60high-granulocyte/stardist/resnet-unet/F/004/015_granulocyte',
    ]
    _train_eval_pretrained_models('dsb2018', configs.data.dsb2018,
                                  'stardist', configs.head.stardist,
                                  'resnet-unet', configs.backbone.resnet_unet,
                                  configs.training.default,
                                  configs.evaluation.instance_segm,
                                  num_train_options,
                                  model_names_pretrained,
                                  args)

    # ------------------------------------------------
    # Frankenstein - Aug
    # ------------------------------------------------
    model_names_pretrained = [
        'T/none/hl60low-granulocyte-aug/stardist/resnet-unet/F/000/020_granulocyte',
        'T/none/hl60low-granulocyte-aug/stardist/resnet-unet/F/001/020_granulocyte',
        'T/none/hl60low-granulocyte-aug/stardist/resnet-unet/F/002/020_granulocyte',
        'T/none/hl60low-granulocyte-aug/stardist/resnet-unet/F/003/020_granulocyte',
        'T/none/hl60low-granulocyte-aug/stardist/resnet-unet/F/004/020_granulocyte',
    ]
    _train_eval_pretrained_models('dsb2018', configs.data.dsb2018,
                                  'stardist', configs.head.stardist,
                                  'resnet-unet', configs.backbone.resnet_unet,
                                  configs.training.default,
                                  configs.evaluation.instance_segm,
                                  num_train_options,
                                  model_names_pretrained,
                                  args)

    # ------------------------------------------------
    # ImageNet
    # ------------------------------------------------
    num_experiments = 3
    _train_eval_imagenet_init_models('dsb2018', configs.data.dsb2018,
                                     'stardist', configs.head.stardist,
                                     'imagenet-resnet-unet', configs.backbone.imagenet_resnet_unet,
                                     configs.training.small_bs_less_cp,
                                     configs.evaluation.instance_segm,
                                     num_train_options + ['F'],
                                     num_experiments,
                                     args)


def _train_eval_random_init_models(name_data, conf_data,
                                   name_head, conf_head,
                                   name_backbone, conf_backbone,
                                   conf_training, conf_eval,
                                   num_train_options, num_experiments,
                                   args):
    for experiment_id in range(num_experiments):
        for num_train in num_train_options:
            _train_eval_random_init_model(name_data, conf_data,
                                          name_head, conf_head,
                                          name_backbone, conf_backbone,
                                          conf_training, conf_eval,
                                          num_train, experiment_id, args)


def _train_eval_random_init_model(name_data, conf_data,
                                  name_head, conf_head,
                                  name_backbone, conf_backbone,
                                  conf_training, conf_eval,
                                  num_train, experiment_id, args):
    max_epochs = 1000
    input_shape = [None, None, 1]
    model_name = _get_model_name_random(name_data, name_head, name_backbone, num_train,
                                        experiment_id)
    conf_data_limited = dict(conf_data)
    if num_train != 'F':
        conf_data_limited['num_train'] = num_train
    conf = {
        'name': model_name,
        'input_shape': input_shape,
        'data': conf_data_limited,
        'head': conf_head,
        'backbone': conf_backbone,
        'training': conf_training,
        'evaluation': conf_eval
    }

    _train_and_evaluate(conf, max_epochs, args)


def _train_eval_imagenet_init_models(name_data, conf_data,
                                     name_head, conf_head,
                                     name_backbone, conf_backbone,
                                     conf_training, conf_eval,
                                     num_train_options, num_experiments,
                                     args):
    for experiment_id in range(num_experiments):
        for num_train in num_train_options:
            _train_eval_imagenet_init_model(name_data, conf_data,
                                            name_head, conf_head,
                                            name_backbone, conf_backbone,
                                            conf_training, conf_eval,
                                            num_train, experiment_id, args)


def _train_eval_imagenet_init_model(name_data, conf_data,
                                    name_head, conf_head,
                                    name_backbone, conf_backbone,
                                    conf_training, conf_eval,
                                    num_train, experiment_id, args):
    max_epochs = 1000
    input_shape = [None, None, 1]
    model_name = _get_model_name_imagenet(name_data, name_head, name_backbone, num_train,
                                          experiment_id)
    conf_data_limited = dict(conf_data)
    if num_train != 'F':
        conf_data_limited['num_train'] = num_train
    conf = {
        'name': model_name,
        'input_shape': input_shape,
        'data': conf_data_limited,
        'head': conf_head,
        'backbone': conf_backbone,
        'training': conf_training,
        'evaluation': conf_eval
    }

    _train_and_evaluate(conf, max_epochs, args)


def _train_eval_pretrained_models(name_data, conf_data,
                                  name_head, conf_head,
                                  name_backbone, conf_backbone,
                                  conf_training, conf_eval,
                                  num_train_options, model_names_pretrained,
                                  args):
    for experiment_id, model_name_pre in enumerate(model_names_pretrained):
        for num_train in num_train_options:
            _train_eval_pretrained_model(name_data, conf_data,
                                         name_head, conf_head,
                                         name_backbone, conf_backbone,
                                         conf_training, conf_eval,
                                         num_train, model_name_pre,
                                         experiment_id, args)


def _train_eval_pretrained_model(name_data, conf_data,
                                 name_head, conf_head,
                                 name_backbone, conf_backbone,
                                 conf_training, conf_eval,
                                 num_train, model_name_pre,
                                 experiment_id, args):
    max_epochs = 1000
    input_shape = [None, None, 1]

    conf_backbone_pretrained = dict(conf_backbone)
    conf_backbone_pretrained['weights'] = os.path.join(
        'models', model_name_pre, 'weights_final.h5')
    name_pre_data = model_name_pre.split('/')[2]

    conf_data_limited = dict(conf_data)
    if num_train != 'F':
        conf_data_limited['num_train'] = num_train

    model_name = _get_model_name_pretrained(name_pre_data, name_data, name_head, name_backbone,
                                            num_train, experiment_id)

    conf = {
        'name': model_name,
        'input_shape': input_shape,
        'data': conf_data_limited,
        'head': conf_head,
        'backbone': conf_backbone_pretrained,
        'training': conf_training,
        'evaluation': conf_eval
    }

    _train_and_evaluate(conf, max_epochs, args)


def _train_frankenstein_models(name_pre_data, conf_datas,
                               name_head, conf_head,
                               name_backbone, conf_backbone,
                               conf_training, conf_eval,
                               epochs_per_model, num_experiments,
                               args):
    for experiment_id in range(num_experiments):
        _train_frankenstein_model(name_pre_data, conf_datas,
                                  name_head, conf_head,
                                  name_backbone, conf_backbone,
                                  conf_training, conf_eval,
                                  epochs_per_model, experiment_id,
                                  args)


def _train_frankenstein_model(name_pre_data, conf_datas,
                              name_head, conf_head,
                              name_backbone, conf_backbone,
                              conf_training, conf_eval,
                              epochs_per_model, experiment_id,
                              args):
    input_shape = [None, None, 1]
    conf_backbone_pretrained = dict(conf_backbone)

    for run_id, (name_data_current, conf_data) in enumerate(conf_datas, 1):

        model_name = _get_model_name_transfer(name_pre_data, name_data_current, name_head,
                                              name_backbone, experiment_id, run_id)

        conf = {
            'name': model_name,
            'input_shape': input_shape,
            'data': conf_data,
            'head': conf_head,
            'backbone': conf_backbone_pretrained,
            'training': conf_training,
            'evaluation': conf_eval
        }

        _train_model(conf, epochs_per_model, args)

        conf_backbone_pretrained['weights'] = os.path.join(
            'models', model_name, 'weights_final.h5')


def _get_model_name_random(name_data, name_head, name_backbone, num_train, experiment_id):
    return 'R/none/{}/{}/{}/{}/{:03d}'.format(name_data, name_head, name_backbone,
                                              _format_num_train(num_train), experiment_id)


def _get_model_name_imagenet(name_data, name_head, name_backbone, num_train, experiment_id):
    return 'P/imagenet/{}/{}/{}/{}/{:03d}'.format(name_data, name_head, name_backbone,
                                                  _format_num_train(num_train), experiment_id)


def _get_model_name_pretrained(name_pre_data, name_data, name_head, name_backbone, num_train,
                               experiment_id):
    return 'P/{}/{}/{}/{}/{}/{:03d}'.format(name_pre_data, name_data, name_head, name_backbone,
                                            _format_num_train(num_train), experiment_id)


def _get_model_name_transfer(name_pre_data, name_data, name_head, name_backbone,
                             experiment_id, run_id):
    return 'T/none/{}/{}/{}/F/{:03d}/{:03d}_{}'.format(name_pre_data, name_head, name_backbone,
                                                       experiment_id, run_id, name_data)


def _format_num_train(num_train):
    if num_train == 'F':
        return 'F'
    return '{:03d}'.format(int(num_train))


def _train_and_evaluate(conf, epochs, args):
    _train_model(conf, epochs, args)
    if not args.no_eval and not args.check:
        _evaluate_model(conf['name'], args)


def _train_model(conf, epochs, args):
    model_name = conf['name']
    print('INFO: Training model {} for {} epochs...'.format(
        model_name, epochs))

    if args.dry_run:
        # Dry run do not train
        if args.verbose:
            print(conf)
        return

    if args.check:
        if os.path.isfile(os.path.join('models', model_name, 'config.yaml')):
            other_conf = utils.utils.yaml_load(
                os.path.join('models', model_name, 'config.yaml'))
            if other_conf != conf:
                print(
                    'ERROR: Model config different.'.format(model_name))
                for k in conf.keys():
                    if conf[k] != other_conf[k]:
                        print('ERROR: Different key: {}'.format(k))
            else:
                print('INFO: Model is fine.')
        else:
            print('ERROR: Model does not exist.')

        return

    if not os.path.isdir(os.path.join('models', model_name)):
        try:
            # Train the model
            train.train(conf, epochs=epochs)
        except Exception as e:
            print('ERROR: Training of model {} failed: {}'.format(
                model_name, e))
            traceback.print_tb(e.__traceback__)
    else:
        # Check if the model has the same config
        other_conf = utils.utils.yaml_load(
            os.path.join('models', model_name, 'config.yaml'))
        if other_conf != conf:
            print(
                'ERROR: Model {} already present but with other config.'.format(model_name))
        else:
            print('INFO: Model {} already present.'.format(model_name))


def _evaluate_model(name, args):
    print('INFO: Evaluating model {}...'.format(name))
    if args.dry_run:
        return

    try:
        results_file = os.path.join('models', name, 'results.csv')
        # Check if the results file already exists
        if os.path.isfile(results_file):
            print('INFO: Model {} already evaluated.'.format(name))
            return

        with open(os.path.join('models', name, 'config.yaml'), 'r') as f:
            conf = yaml_load(f)

        # Evaluate all epochs
        results = {}
        # TODO set start epoch to 0
        epoch = utils.utils.get_last_epoch(os.path.join('models', name))
        while True:
            try:
                print("Epoch {}".format(epoch))
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
        results_df = results_df.set_index('epoch')
        results_df.to_csv(results_file)
        print("INFO: Done evaluating model {}".format(name))
    except Exception as e:
        print('ERROR: Evaluation of model {} failed: {}'.format(name, e))
        traceback.print_tb(e.__traceback__)


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
