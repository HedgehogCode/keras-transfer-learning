import os

import numpy as np

from .config import DataConfig
from .data import stardist_dsb2018, datagen


class StarDistDSB2018DataConfig(DataConfig):

    def __init__(self, train_val_split, training_size, normalizer,
                 data_dir=os.path.join('data', 'stardist-dsb2018')):
        self._train_val_split = train_val_split
        self._data_dir = data_dir
        self._training_size = training_size
        self._normalizer = normalizer
        self._data = None

    def load_data(self, seed):
        train_x, train_y, val_x, val_y = stardist_dsb2018.load_train(
            data_dir=self._data_dir, seed=seed, train_val_split=self._train_val_split)
        self._data = {
            'train_x': train_x,
            'train_y': train_y,
            'val_x': val_x,
            'val_y': val_y
        }

    def _normalize_prepare(self, prepare_fn):
        def apply(batch_x, batch_y):
            return prepare_fn(self._normalizer(np.array(batch_x)), batch_y)
        return apply

    def create_train_datagen(self, batch_size, prepare_fn, seed):
        # TODO make data augmentation configurable
        dataaug_fn = datagen.dataaug_fn_crop_flip_2d(
            self._training_size[0], self._training_size[1])

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
