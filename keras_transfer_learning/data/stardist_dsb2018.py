import os

from .utils import load_and_split, load_images_and_masks


def load_train(data_dir=os.path.join('data', 'stardist-dsb2018'), seed=42, train_val_split=0.9):
    train_dir = os.path.join(data_dir, 'train')
    return load_and_split(train_dir, seed=seed, train_val_split=train_val_split)


def load_test(data_dir='data/stardist-dsb2018'):
    test_dir = os.path.join(data_dir, 'test')
    return load_images_and_masks(test_dir)
