import struct

import numpy as np
import keras


class DataGenerator(keras.utils.Sequence):
    """A generic keras data generator. Handle shuffeling, reproducability and batches."""

    def __init__(self, ids, batch_size, data_fn, shuffle=True, epoch_len=None):
        self.ids = ids
        self.batch_size = batch_size

        self.data_fn = data_fn

        self.shuffle = shuffle
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
        # Prepare the indexes for this epoch
        self.indexes = np.arange(len(self.ids))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Indices of this batch
        idxs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        ids = [self.ids[k] for k in idxs]

        # Apply data augmentation
        batch_x, batch_y = self.data_fn(ids)

        return batch_x, batch_y


def data_generator_from_lists(batch_size, data_x, data_y, dataaug_fn, prepare_fn, **kwargs):
    ids = list(range(len(data_x)))
    data_fn = data_fn_from_lists(data_x, data_y, dataaug_fn, prepare_fn)
    return DataGenerator(ids, batch_size, data_fn, **kwargs)


def data_generator_for_validation(val_x, val_y, prepare_fn):
    ids = list(range(len(val_x)))
    data_fn = data_fn_from_lists(
        val_x, val_y, lambda x, y: (x, y), prepare_fn)
    return DataGenerator(ids, 1, data_fn, shuffle=False)


def data_fn_from_lists(data_x, data_y, dataaug_fn, prepare_fn):
    def data_fn(ids):
        # Loading
        batch_x = [data_x[id] for id in ids]
        batch_y = [data_y[id] for id in ids]

        # Dataaug
        batch_x, batch_y = dataaug_fn(batch_x, batch_y)

        # Prepare for training
        return prepare_fn(batch_x, batch_y)
    return data_fn
