import os

from .utils import load_and_split, load_images_and_masks


def load_train(data_dir, **kwargs):
    train_dir = os.path.join(data_dir, 'train')
    return load_and_split(train_dir, **kwargs)


def load_test(data_dir):
    test_dir = os.path.join(data_dir, 'test')
    return load_images_and_masks(test_dir)
