import os
import glob
import numpy as np

import tifffile

from keras_transfer_learning.heads import segm, stardist, classification
from keras_transfer_learning.data import stardist_dsb2018, dsb2018, cytogen, cityscapes
from keras_transfer_learning.data import dataaug, datagen, utils


def _create_data_generators(conf):
    # Decide which function is appropriate
    return {
        'stardist-dsb2018': _create_data_generators_from_lists,
        'dsb2018': _create_data_generators_from_lists,
        'cytogen': _create_data_generators_from_lists,
        'cityscapes': _create_data_generators_from_lists
    }[conf['data']['name']](conf)


def _create_data_generators_from_lists(conf):
    # Find the appropriate load function
    load_fn = {
        'stardist-dsb2018': stardist_dsb2018.load_train,
        'dsb2018': dsb2018.load_train,
        'cytogen': cytogen.load_train,
        'cityscapes': cityscapes.load_train
    }[conf['data']['name']]

    # Load the data
    seed = conf['data'].get('datasplit_seed', 42)
    train_val_split = conf['data'].get('train_val_split', 0.9)
    num_train = conf['data'].get('num_train', None)
    part = conf['data'].get('part', 0)
    train_x, train_y, val_x, val_y = load_fn(
        data_dir=conf['data']['data_dir'], seed=seed, train_val_split=train_val_split,
        num_train=num_train, part=part)

    # Normalizer each sample
    normalize_fn = _create_normalize_fn(conf)
    if conf['data']['name'] in ['stardist-dsb2018', 'dsb2018', 'cytogen']:
        train_x = [normalize_fn(x) for x in train_x]
        val_x = [normalize_fn(x) for x in val_x]
        normalize_fn = None

    # Create the prepare and dataaug functions
    prepare_fn = _create_prepare_fn(conf)
    dataaug_fn = _create_dataaug_fn(conf)

    # Create the generators
    train_gen = datagen.data_generator_from_lists(
        batch_size=conf['training']['batch_size'], data_x=train_x, data_y=train_y,
        dataaug_fn=dataaug_fn, prepare_fn=prepare_fn, normalize_fn=normalize_fn,
        epoch_len=conf['training']['epoch_length'])
    val_gen = datagen.data_generator_for_validation(
        val_x=val_x, val_y=val_y, prepare_fn=prepare_fn, normalize_fn=normalize_fn)

    return train_gen, val_gen


def _create_normalize_fn(conf):
    return {
        'uint8-range': lambda x: x / 255,
        'min-max': lambda x: (x - np.min(x)) / np.max(x)
    }[conf['data']['normalizer']]


def _create_dataaug_fn(conf):
    return {
        'imgaug': lambda: dataaug.create_imgaug_augmentor(conf['data']['dataaug']['augmentors']),
    }[conf['data']['dataaug']['name']]()


def _create_prepare_fn(conf):
    return {
        'stardist': lambda x, y: stardist.prepare_data(conf['head']['args']['n_rays'], x, y),
        'segm': lambda x, y: segm.prepare_data_nclass(x, y, conf['head']['num_classes']),
        'fgbg-segm': segm.prepare_data_fgbg,
        'fgbg-segm-weighted': lambda x, y: segm.prepare_data_fgbg_weigthed(
            batch_x=x, batch_y=y,
            **conf['head']['prepare_data_args']),
        'classification': classification.prepare_data
    }[conf['head']['name']]


class Dataset():

    def __init__(self, config):
        self.config = config

    def create_data_generators(self):
        return _create_data_generators(self.config)

    def get_random_test_img(self, seed=None):
        data_dir = os.path.join(self.config['data']['data_dir'], 'test')
        img_dir = os.path.join(data_dir, 'images')
        mask_dir = os.path.join(data_dir, 'masks')

        img_files = sorted(glob.glob(os.path.join(img_dir, '*.tif')))
        mask_files = sorted(glob.glob(os.path.join(mask_dir, '*.tif')))

        idx = np.random.RandomState(seed).choice(len(img_files))

        img = tifffile.imread(img_files[idx])
        mask = tifffile.imread(mask_files[idx])

        # Normalize the image
        img = _create_normalize_fn(self.config)(img)

        return img, mask

    def create_test_dataset(self):
        data_dir = os.path.join(self.config['data']['data_dir'], 'test')
        imgs, masks = utils.load_images_and_masks(data_dir)

        normalize_fn = _create_normalize_fn(self.config)
        normalized_imgs = [normalize_fn(img) for img in imgs]
        return normalized_imgs, masks
