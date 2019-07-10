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


MODEL_NAME_MAP = {}


def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-f', '--filter', type=str, default='*')
    args = parser.parse_args(arguments)
    filter_glob = args.filter

    model_dirs = sorted(glob.glob(os.path.join('models', filter_glob)))

    if os.path.isfile('model_map.yaml'):
        with open('model_map.yaml', 'r') as f:
            MODEL_NAME_MAP.update(yaml_load(f))

    # Rename the models
    for m in model_dirs:
        auto_rename_model(m)

    print(MODEL_NAME_MAP)
    with open('model_map.yaml', 'w') as f:
        yaml.dump(MODEL_NAME_MAP, f)

    # Fix the weight paths
    new_model_dirs = [os.path.join('models', n)
                      for n in MODEL_NAME_MAP.values()]
    for m in new_model_dirs:
        fix_weigths_path(m)


def auto_rename_model(model_dir):
    model_name_old = model_dir.split('/')[-1]

    name_experiment, name_backbone, name_head, \
        name_data, pretrained, num_train = model_name_old.split('_')


    # ----------- Pretrain experiments + monster model
    if name_experiment[0] in 'ABCDEFGI':
        name_data_new = _map_data_name(name_data)

        if pretrained == 'R':
            model_name_new = 'R/none/{}/{}/{}/{}/'.format(name_data_new, name_head,
                                                          name_backbone, num_train)
        elif pretrained == 'P':
            # Find the data name of the weights model
            conf = _read_config(model_dir)
            weights_model = conf['backbone']['weights'].split('/')[1]
            weights_data = _map_data_name(weights_model.split('_')[3])

            model_name_new = 'P/{}/{}/{}/{}/{}/'.format(weights_data, name_data_new, name_head,
                                                        name_backbone, num_train)
        else:
            raise ValueError('Unkown pretrained type: {}'.format(pretrained))

        print('Automatic: "{}" -> "{}"'.format(model_name_old, model_name_new))
        # Rename the model
        exp_id = 0
        while True:
            try:
                _rename_model(model_name_old,
                            model_name_new + '{:03d}'.format(exp_id))
                break
            except ValueError:
                exp_id += 1

    # ------------- Frankenstein H
    elif name_experiment[0] == 'H':
        experiment_id = int(name_experiment[1:])
        if pretrained == 'R':
            # One of the transfer learning models
            name_data_new = _map_data_name(name_data[:-2])
            run_id = int(name_data[-2:])
            prefix = 'T/none/hl60low-hl60high-granulocyte/stardist/resnet-unet/F/'
            model_name_new = prefix + '{:03d}/{:03d}_{}'.format(experiment_id, run_id, name_data_new)
        elif pretrained == 'P':
            # One of the final models
            prefix = 'P/hl60low-hl60high-granulocyte/dsb2018/stardist/resnet-unet/'
            model_name_new = prefix + '{}/{:03d}'.format(num_train, experiment_id)
        else:
            raise ValueError('Unkown pretrained type: {}'.format(pretrained))

        print('Automatic: "{}" -> "{}"'.format(model_name_old, model_name_new))
        _rename_model(model_name_old, model_name_new)


    # ------------- Frankenstein J
    elif name_experiment[0] == 'J':
        experiment_id = int(name_experiment[1:])
        if pretrained == 'R':
            # One of the transfer learning models
            name_data_new = _map_data_name(name_data[:-3])
            run_id = int(name_data[-2:])
            prefix = 'T/none/hl60low-granulocyte-aug/stardist/resnet-unet/F/'
            model_name_new = prefix + '{:03d}/{:03d}_{}'.format(experiment_id, run_id, name_data_new)
        elif pretrained == 'P':
            # One of the final models
            prefix = 'P/hl60low-granulocyte-aug/dsb2018/stardist/resnet-unet/'
            model_name_new = prefix + '{}/{:03d}'.format(num_train, experiment_id)
        else:
            raise ValueError('Unkown pretrained type: {}'.format(pretrained))

        print('Automatic: "{}" -> "{}"'.format(model_name_old, model_name_new))
        _rename_model(model_name_old, model_name_new)


    # ------------- Manual
    else:
        model_name_new = None
        while model_name_new is None:
            print('Model name: "{}"'.format(model_name_old))
            print('Manually setting the model name.')
            model_name_new = input('New model name: ')

            print('Manual: "{}" -> "{}"'.format(model_name_old, model_name_new))
            accept = input('(ACCEPT/reject)? >')
            if accept not in ['', 'ACCEPT', 'accept']:
                model_name_new = None

        # Rename the model
        _rename_model(model_name_old, model_name_new)


def fix_weigths_path(model_dir):
    print('Fix weight path for "{}"...'.format(model_dir))
    conf = _read_config(model_dir)
    weights_path_old = conf['backbone']['weights']


    if weights_path_old is not None:
        if weights_path_old.startswith('models/'):
            # Already fixed
            return
        if weights_path_old.startswith('models2/'):
            weights_path_new = 'models' + weights_path_old[7:]
        else:
            model_name_old = weights_path_old.split('/')[1]
            model_name_new = MODEL_NAME_MAP[model_name_old]
            weights_path_new = os.path.join(
                'models2', model_name_new, weights_path_old.split('/')[-1])

        conf['backbone']['weights'] = weights_path_new
        _write_config(model_dir, conf)


def _read_config(model_dir):
    with open(os.path.join(model_dir, 'config.yaml'), 'r') as f:
        return yaml_load(f)


def _write_config(model_dir, conf):
    with open(os.path.join(model_dir, 'config.yaml'), 'w') as f:
        yaml.dump(conf, f)


def _rename_model(model_name_old, model_name_new):
    model_dir_old = os.path.join('models', model_name_old)
    model_dir_new = os.path.join('models2', model_name_new)

    log_dir_old = os.path.join('logs', model_name_old)
    log_dir_new = os.path.join('logs2', model_name_new)

    if os.path.isdir(model_dir_new):
        raise ValueError(
            'The target model "{}" already exists.'.format(model_dir_new))
    if os.path.isdir(log_dir_new):
        raise ValueError(
            'The target log directory "{}" already exists.'.format(log_dir_new))

    # Save to the model map
    MODEL_NAME_MAP[model_name_old] = model_name_new

    # Load the config
    conf = _read_config(model_dir_old)
    conf['name'] = model_name_new

    # Copy the model dir
    shutil.copytree(model_dir_old, model_dir_new)

    # Copy the log dir
    shutil.copytree(log_dir_old, log_dir_new)

    # Remove the backup files
    backup_files = glob.glob(os.path.join(model_dir_new, '*.bak*')) + \
        glob.glob(os.path.join(model_dir_new, '*.copy'))
    for backup_file in backup_files:
        os.remove(backup_file)

    # Save the new config
    _write_config(model_dir_new, conf)


def _map_data_name(data_name):
    if data_name == 'hl60':
        return 'hl60low'
    if data_name == 'dsb2018':
        return 'dsb2018'
    if data_name == 'hl60-low-noise':
        return 'hl60low'
    if data_name == 'hl60-high-noise':
        return 'hl60high'
    if data_name == 'granulocyte':
        return 'granulocyte'
    if data_name == 'cityscapes':
        return 'cityscapes'
    raise ValueError('Unknown data: {}'.format(data_name))


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
