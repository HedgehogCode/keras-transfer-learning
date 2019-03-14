import os
from abc import abstractmethod

import numpy as np

from .config_holder import ConfigHolder
from ..data import stardist_dsb2018, datagen


# Abstract definition
class DataConfig(ConfigHolder):

    @abstractmethod
    def load_data(self, seed):
        raise NotImplementedError

    @abstractmethod
    def create_train_datagen(self, batch_size, prepare_fn, seed):
        raise NotImplementedError

    @abstractmethod
    def create_val_datagen(self, prepare_fn):
        raise NotImplementedError


class Normalizer(ConfigHolder):

    @abstractmethod
    def __call__(self, data):
        raise NotImplementedError


# Static config parser

def get_config(conf) -> DataConfig:
    # Define the normalizer
    if conf['normalizer'] == 'uint8-range':
        normalizer = UInt8RangeNormalizer()
    else:
        raise NotImplementedError(
            'The normalizer {} is not implemented.'.format(conf['normalizer']))

    # Define the data config
    if conf['name'] == 'stardist-dsb2018':
        return StarDistDSB2018DataConfig(
            train_val_split=conf['train_val_split'],
            training_size=conf['training_size'],
            normalizer=normalizer)
    raise NotImplementedError(
        'The data {} is not implemented.'.format(conf['name']))


# Implementations

class UInt8RangeNormalizer(Normalizer):

    def __call__(self, data):
        return data / 255

    def get_as_dict(self):
        return 'uint8-range'


class StarDistDSB2018DataConfig(DataConfig):

    def __init__(self, train_val_split, training_size, normalizer,
                 data_dir=os.path.join('data', 'stardist-dsb2018')):
        self.train_val_split = train_val_split
        self.data_dir = data_dir
        self.training_size = training_size
        self.normalizer = normalizer
        self._data = None

    def load_data(self, seed):
        train_x, train_y, val_x, val_y = stardist_dsb2018.load_train(
            data_dir=self.data_dir, seed=seed, train_val_split=self.train_val_split)
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

    def create_train_datagen(self, batch_size, prepare_fn, seed):
        # TODO make data augmentation configurable
        dataaug_fn = datagen.dataaug_fn_crop_flip_2d(
            self.training_size[0], self.training_size[1])

        return datagen.data_generator_from_lists(
            batch_size=batch_size,
            data_x=self._data['train_x'],
            data_y=self._data['train_y'],
            dataaug_fn=dataaug_fn,
            prepare_fn=self._normalize_prepare(prepare_fn),
            seed=seed)

    def create_val_datagen(self, prepare_fn):
        return datagen.data_generator_for_validation(
            val_x=self._data['val_x'],
            val_y=self._data['val_y'],
            prepare_fn=self._normalize_prepare(prepare_fn))

    def get_as_dict(self):
        return {
            'name': 'stardist-dsb2018',
            'train_val_split': self.train_val_split,
            'training_size': self.training_size,
            'data_dir': self.data_dir,
            'normalizer': self.normalizer.get_as_dict()
        }
