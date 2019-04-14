import os
import math
from glob import glob

import numpy as np
from tifffile import imread


def load_and_split(data_dir: str, seed: int = 42, train_val_split: float = 0.9):
    """Loads images and masks from a directory and splits them into two datasets (train and val).
    The directory should contain two directories 'images' and 'masks' which should contian tiff
    files with the same names.

    Arguments:
        data_dir {str} -- path of the data directory

    Keyword Arguments:
        seed {int} -- seed for the train/val split (default: {42})
        train_val_split {float} -- relation of the train/val split (default: {0.9})

    Returns:
        tuple -- x_train, y_train, x_val, y_val
    """

    X, Y = load_images_and_masks(data_dir)

    # Train/Validation split
    num = len(X)
    num_train = math.floor(num * train_val_split)
    random_idxs = np.random.RandomState(seed).permutation(num)
    train_idxs = random_idxs[:num_train]
    val_idxs = random_idxs[num_train:]

    def get(A, idxs):
        return [A[i] for i in idxs]

    return get(X, train_idxs), get(Y, train_idxs), get(X, val_idxs), get(Y, val_idxs)


def load_images_and_masks(folder):
    images_dir = os.path.join(folder, 'images')
    masks_dir = os.path.join(folder, 'masks')
    images = _load_tifs_from_dir_sorted(images_dir)
    masks = _load_tifs_from_dir_sorted(masks_dir)
    return images, masks


def _load_tifs_from_dir_sorted(folder):
    files = sorted(glob(os.path.join(folder, '*.tif')))
    return list(map(imread, files))
