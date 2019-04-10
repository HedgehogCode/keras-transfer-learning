import os
from abc import abstractmethod

import numpy as np

from imgaug import augmenters as iaa

from .config_holder import ConfigHolder
from ..data import stardist_dsb2018, datagen, cytogen


# Abstract definition
class DataConfig(ConfigHolder):

    @abstractmethod
    def load_data(self):
        raise NotImplementedError

    @abstractmethod
    def create_train_datagen(self, batch_size, prepare_fn):
        raise NotImplementedError

    @abstractmethod
    def create_val_datagen(self, prepare_fn):
        raise NotImplementedError


class Normalizer(ConfigHolder):

    @abstractmethod
    def __call__(self, data):
        raise NotImplementedError


class DataAugmenter(ConfigHolder):

    @abstractmethod
    def __call__(self, data):
        raise NotImplementedError


# Static config parser

def get_config(conf) -> DataConfig:
    # Define the normalizer
    if conf['normalizer'] == 'uint8-range':
        normalizer = UInt8RangeNormalizer()
    if conf['normalizer'] == 'min-max':
        normalizer = MinMaxNormalizer()
    else:
        raise NotImplementedError(
            'The normalizer {} is not implemented.'.format(conf['normalizer']))

    # Define the augmentor
    conf_dataaug = conf['dataaug']
    if conf_dataaug['name'] == 'imgaug':
        dataaug = ImageAugDataAugmenter(conf_dataaug['augmentors'])
    else:
        raise NotImplementedError(
            'The data augmentor {} is not implemented.'.format(conf_dataaug['name']))

    # Define the data config
    if conf['name'] == 'stardist-dsb2018':
        return StarDistDSB2018DataConfig(
            train_val_split=conf['train_val_split'],
            datasplit_seed=conf['datasplit_seed'],
            dataaug=dataaug,
            normalizer=normalizer)
    if conf['name'] == 'cytogen':
        return CytogenDataConfig(
            data_dir=conf['data_dir'],
            train_val_split=conf['train_val_split'],
            datasplit_seed=conf['datasplit_seed'],
            dataaug=dataaug,
            normalizer=normalizer)
    raise NotImplementedError(
        'The data {} is not implemented.'.format(conf['name']))


# Implementations

class UInt8RangeNormalizer(Normalizer):

    def __call__(self, data):
        return data / 255

    def get_as_dict(self):
        return 'uint8-range'


class MinMaxNormalizer(Normalizer):

    def __call__(self, data):
        return data - np.min(data) / np.max(data)

    def get_as_dict(self):
        return 'min-max'


class ImageAugDataAugmenter(DataAugmenter):

    def __init__(self, augmentors):
        self.augmentors = augmentors
        augs = []
        for aug in self.augmentors:
            augs.append(getattr(iaa, aug['name'])(**aug['args']))
        self.augmentor = iaa.Sequential(augs)

    def __call__(self, data):
        aug_det = self.augmentor.to_deterministic()
        augmented = []
        for batch in data:
            augmented.append(aug_det.augment_images(batch))
        return augmented

    def get_as_dict(self):
        return {
            'name': 'imgaug',
            'augmentors': self.augmentors
        }


class StarDistDSB2018DataConfig(DataConfig):

    def __init__(self, train_val_split, datasplit_seed, dataaug, normalizer,
                 data_dir=os.path.join('data', 'stardist-dsb2018')):
        self.train_val_split = train_val_split
        self.datasplit_seed = datasplit_seed
        self.data_dir = data_dir
        self.dataaug = dataaug
        self.normalizer = normalizer

        self._data = None

    def load_data(self):
        train_x, train_y, val_x, val_y = stardist_dsb2018.load_train(
            data_dir=self.data_dir, seed=self.datasplit_seed, train_val_split=self.train_val_split)
        self._data = {
            'train_x': train_x,
            'train_y': train_y,
            'val_x': val_x,
            'val_y': val_y
        }

    def _normalize_prepare(self, prepare_fn):
        def apply(batch_x, batch_y):
            return prepare_fn(self.normalizer(np.array(batch_x)), batch_y)
        return apply

    def create_train_datagen(self, batch_size, prepare_fn):
        return datagen.data_generator_from_lists(
            batch_size=batch_size,
            data_x=self._data['train_x'],
            data_y=self._data['train_y'],
            dataaug_fn=self.dataaug,
            prepare_fn=self._normalize_prepare(prepare_fn))

    def create_val_datagen(self, prepare_fn):
        return datagen.data_generator_for_validation(
            val_x=self._data['val_x'],
            val_y=self._data['val_y'],
            prepare_fn=self._normalize_prepare(prepare_fn))

    def get_as_dict(self):
        return {
            'name': 'stardist-dsb2018',
            'train_val_split': self.train_val_split,
            'data_dir': self.data_dir,
            'dataaug': self.dataaug.get_as_dict(),
            'normalizer': self.normalizer.get_as_dict(),
            'datasplit_seed': self.datasplit_seed
        }


class CytogenDataConfig(DataConfig):

    def __init__(self, data_dir, train_val_split, datasplit_seed, dataaug, normalizer):
        self.data_dir = data_dir
        self.train_val_split = train_val_split
        self.datasplit_seed = datasplit_seed
        self.dataaug = dataaug
        self.normalizer = normalizer

        self._data = None

    def load_data(self):
        train_x, train_y, val_x, val_y = cytogen.load_train(
            data_dir=self.data_dir, seed=self.datasplit_seed, train_val_split=self.train_val_split)
        self._data = {
            'train_x': train_x,
            'train_y': train_y,
            'val_x': val_x,
            'val_y': val_y
        }

    def _normalize_prepare(self, prepare_fn):
        def apply(batch_x, batch_y):
            return prepare_fn(self.normalizer(np.array(batch_x)), batch_y)
        return apply

    def create_train_datagen(self, batch_size, prepare_fn):
        return datagen.data_generator_from_lists(
            batch_size=batch_size,
            data_x=self._data['train_x'],
            data_y=self._data['train_y'],
            dataaug_fn=self.dataaug,
            prepare_fn=self._normalize_prepare(prepare_fn))

    def create_val_datagen(self, prepare_fn):
        return datagen.data_generator_for_validation(
            val_x=self._data['val_x'],
            val_y=self._data['val_y'],
            prepare_fn=self._normalize_prepare(prepare_fn))

    def get_as_dict(self):
        return {
            'name': 'cytogen',
            'data_dir': self.data_dir,
            'train_val_split': self.train_val_split,
            'dataaug': self.dataaug.get_as_dict(),
            'normalizer': self.normalizer.get_as_dict(),
            'datasplit_seed': self.datasplit_seed
        }
