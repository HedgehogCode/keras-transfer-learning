"""General utilities
"""

import os
import re
import glob
from yaml import unsafe_load

from keras import models

WEIGHTS_FILE_REGEX = r'.*_(\d{4})_\d+\.\d{4}\.h5'

def get_last_weights(model_dir: str, epoch: int = None):
    """Finds the filename of the last weights file in a model directory.

    Arguments:
        model_dir {str} -- path to the model directory

    Keyword Arguments:
        epoch {int} -- The expected last epoch. If another epoch is the last a ValueError will be
                       raised. (default: {None})

    Raises:
        ValueError -- If no weights file is found or the last weights file is of another epoch

    Returns:
        {str} -- The path to the last weights file.
    """

    weights = sorted(glob.glob(os.path.join(model_dir, 'weights_[0-9]*_*.h5')))
    if weights == []:
        raise ValueError('Did not find a valid weights file.')
    last_weights = weights[-1]
    matches = re.match(WEIGHTS_FILE_REGEX, last_weights)
    if matches is None:
        raise ValueError('Did not find a valid weights file.')

    # Check if the last epoch is expected
    if epoch is not None:
        last_epoch = int(matches.group(1))
        if last_epoch != epoch:
            raise ValueError('Cannot continue with after epoch {}. Last epoch was {}.'.format(
                epoch, last_epoch))

    return last_weights


def get_epoch_weights(model_dir: str, epoch: int):
    """Finds the filename of the weights file for the given epoch in a model directory.

    Arguments:
        model_dir {str} -- path to the model directory
        epoch {int} -- the epoch

    Raises:
        ValueError -- If no weights file is found for this epoch

    Returns:
        {str} -- The path to the last weights file.
    """

    weights = sorted(glob.glob(os.path.join(model_dir, 'weights_[0-9]*_*.h5')))
    if weights == []:
        raise ValueError('Did not find a valid weights file.')
    for epoch_weights in weights:
        matches = re.match(WEIGHTS_FILE_REGEX, epoch_weights)
        if matches is None:
            raise ValueError('Did not find a valid weights file.')
        if int(matches.group(1)) == epoch:
            return epoch_weights

    raise ValueError('Cannot find weight file for epoch {}'.format(epoch))


def get_last_epoch(model_dir: str):
    """Returns the number of the last epoch of the trained model.

    Arguments:
        model_dir {str} -- path to the model directory

    Raises:
        ValueError -- If no weights files were found or if they are named wrongly.

    Returns:
        {int} -- the number of the last epoch
    """
    weights = sorted(glob.glob(os.path.join(model_dir, 'weights_[0-9]*_*.h5')))
    if weights == []:
        raise ValueError('Did not find a valid weights file.')
    last_weights = weights[-1]
    matches = re.match(WEIGHTS_FILE_REGEX, last_weights)
    if matches is None:
        raise ValueError('Did not find a valid weights file.')

    # Return the epoch of the last weights
    return int(matches.group(1))


def model_up_to_layer(keras_model, layer_name):
    for l in keras_model.layers:
        if l.name == layer_name:
            break
    return models.Model(keras_model.input, l.output)


def yaml_load(file_name: str):
    """Loads the given yaml file from the disc.

    Arguments:
        file_name {str} -- path to the yaml file

    Returns:
        The content of the yaml file
    """
    with open(file_name, 'r') as f:
        content = unsafe_load(f)
    return content


def path_to_model_config(model_name):
    return os.path.join('.', 'models', model_name, 'config.yaml')


def list_model_names(rootdir: str = 'models') -> list:
    model_names = []
    for root, _, files in os.walk(rootdir):
        if 'config.yaml' in files:
            model_names.append(root[7:])
    return sorted(model_names)


def list_model_dirs(rootdir: str = 'models') -> list:
    return [os.path.join(rootdir, n) for n in list_model_names(rootdir)]
