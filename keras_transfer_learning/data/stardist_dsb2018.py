import os

from .utils import load_and_split, load_images_and_masks


def load_train(data_dir=os.path.join('data', 'stardist-dsb2018'), **kwargs):
    train_dir = os.path.join(data_dir, 'train')
    return load_and_split(train_dir, **kwargs)


def load_test(data_dir='data/stardist-dsb2018'):
    test_dir = os.path.join(data_dir, 'test')
    return load_images_and_masks(test_dir)
