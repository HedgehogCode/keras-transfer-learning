import os
import math
from glob import glob

import numpy as np
from tifffile import imread


def load_and_split(data_dir: str, seed: int = 42, train_val_split: float = 0.9,
                   num_train: int = None, part: int = 0):
    """Loads images and masks from a directory and splits them into two datasets (train and val).
    The directory should contain two directories 'images' and 'masks' which should contian tiff
    files with the same names.

    Arguments:
        data_dir {str} -- path of the data directory

    Keyword Arguments:
        seed {int} -- seed for the train/val split (default: {42})
        train_val_split {float} -- relation of the train/val split (default: {0.9})
        num_train {int} -- number of training images to limit the dataset. None to use all.
                           (default: {None})
        part {int} -- which part of the training dataset to use if the dataset was limited.
                      (default: {0})

    Returns:
        tuple -- x_train, y_train, x_val, y_val
    """

    num = len(glob(os.path.join(data_dir, 'images', '*.tif')))
    random_idxs = np.random.RandomState(seed).permutation(num)

    # Full train and val dataset
    num_all_train = math.floor(num * train_val_split)
    train_idxs = random_idxs[:num_all_train]
    val_idxs = random_idxs[num_all_train:]

    if num_train is not None:
        offset = (len(train_idxs) - num_train) // 10
        part = part % 10
        train_idxs = train_idxs[(offset*part):(offset*part)+num_train]

    # Load the data
    x_train, y_train = load_images_and_masks(data_dir, train_idxs)
    x_val, y_val = load_images_and_masks(data_dir, val_idxs)
    return x_train, y_train, x_val, y_val


def load_images_and_masks(folder, ids=None):
    images_dir = os.path.join(folder, 'images')
    masks_dir = os.path.join(folder, 'masks')
    images = _load_tifs_from_dir_sorted(images_dir, ids=ids)
    masks = _load_tifs_from_dir_sorted(masks_dir, ids=ids)
    return images, masks


def _load_tifs_from_dir_sorted(folder, ids=None):
    files = sorted(glob(os.path.join(folder, '*.tif')))
    if ids is not None:
        files = [files[i] for i in ids]
    return list(map(imread, files))
