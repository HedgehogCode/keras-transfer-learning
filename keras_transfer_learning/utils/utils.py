"""General utilities
"""

import os
import re
import glob


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
    matches = re.match(r'.*_(\d{4})_\d+\.\d{4}\.h5', last_weights)
    if matches is None:
        raise ValueError('Did not find a valid weights file.')

    # Check if the last epoch is expected
    if epoch is not None:
        last_epoch = int(matches.group(1))
        if last_epoch != epoch:
            raise ValueError('Cannot continue with after epoch {}. Last epoch was {}.'.format(
                epoch, last_epoch))

    return last_weights
