import struct

import numpy as np
import keras
from imgaug import augmenters as iaa


class DataGenerator(keras.utils.Sequence):

    def __init__(self, ids, batch_size, data_fn, shuffle=True, seed=42, epoch_len=None):
        self.ids = ids
        self.batch_size = batch_size

        self.data_fn = data_fn

        self.shuffle = shuffle
        self.random_state = seed
        # NOTE: epoch_len is the number of samples per epoch not number of batches
        if epoch_len is not None:
            self.epoch_len = epoch_len
        else:
            self.epoch_len = len(ids)
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(self.epoch_len / self.batch_size))

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        # Generate a new seed using the old one
        self.random_state = struct.unpack(
            '<I', np.random.RandomState(self.random_state).bytes(4))[0]

        # Prepare the indexes for this epoch
        self.indexes = np.arange(len(self.ids))
        if self.shuffle is True:
            np.random.RandomState(self.random_state).shuffle(self.indexes)

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Indices of this batch
        idxs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        ids = [self.ids[k] for k in idxs]

        # Apply data augmentation
        dataug_seed = self.random_state * (index + 2)
        x, y = self.data_fn(ids, dataug_seed)

        return x, y


def data_generator_from_lists(batch_size, data_X, data_Y, dataaug_fn, prepare_fn, **kwargs):
    ids = list(range(len(data_X)))
    data_fn = data_fn_from_lists(data_X, data_Y, dataaug_fn, prepare_fn)
    return DataGenerator(ids, batch_size, data_fn, **kwargs)


def data_fn_from_lists(data_X, data_Y, dataaug_fn, prepare_fn):
    def data_fn(ids, seed):
        # Loading
        X = [data_X[id] for id in ids]
        Y = [data_Y[id] for id in ids]

        # Dataaug
        X, Y = dataaug_fn(X, Y, seed)

        # Prepare for training
        return prepare_fn(X, Y)
    return data_fn


def dataug_fn_crop_flip_2d(width, height):
    aug = iaa.Sequential([
        iaa.CropToFixedSize(width, height),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5)
    ])

    def dataaug_fn(X, Y, seed):
        # TODO use seed
        aug_det = aug.to_deterministic()
        X = aug_det.augment_images(X)
        Y = aug_det.augment_images(Y)
        return X, Y
    return dataaug_fn
