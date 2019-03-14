import os
import math
from glob import glob

import numpy as np
from tifffile import imread


def load_train(data_dir=os.path.join('data', 'stardist-dsb2018'), seed=42, train_val_split=0.9):
    train_dir = os.path.join(data_dir, 'train')
    X, Y = _load_images_and_masks(train_dir)

    # Train/Validation split
    num = len(X)
    num_train = math.floor(num * train_val_split)
    random_idxs = np.random.RandomState(seed).permutation(num)
    train_idxs = random_idxs[:num_train]
    val_idxs = random_idxs[num_train:]

    def get(A, idxs):
        return [A[i] for i in idxs]

    return get(X, train_idxs), get(Y, train_idxs), get(X, val_idxs), get(Y, val_idxs)


def load_test(data_dir='data/stardist-dsb2018'):
    test_dir = os.path.join(data_dir, 'test')
    X, Y = _load_images_and_masks(test_dir)
    return X, Y


def _load_images_and_masks(folder):
    images_dir = os.path.join(folder, 'images')
    masks_dir = os.path.join(folder, 'masks')
    images = _load_tifs_from_dir_sorted(images_dir)
    masks = _load_tifs_from_dir_sorted(masks_dir)
    return images, masks


def _load_tifs_from_dir_sorted(folder):
    files = sorted(glob(os.path.join(folder, '*.tif')))
    return list(map(imread, files))
