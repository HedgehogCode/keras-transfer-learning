import struct
import math

import numpy as np
import keras


class DataGenerator(keras.utils.Sequence):
    """A generic keras data generator. Handles shuffeling and batches. Also allowes to set the
    epoch length independend from the number of data samples.
    """

    def __init__(self, ids: list, batch_size: int, data_fn: callable, shuffle: bool = True,
                 epoch_len: int = None):
        """A data generator which can be used to train a keras model with the `fit_generator`
        method.
        Allows to set the nuber of samples per epoch. If the selected number of samples per epoch
        is smaller than the provided number of samples each sample is repeated in one epoch. If the
        selected number of samples is larger than the provided number of samples samples that were
        not shown in one epoch are shown in the next epoch.


        Arguments:
            ids {list} -- list of ids of the data. Selected ids are passed to the `data_fn`.
            batch_size {int} -- size of one data batch
            data_fn {callable} -- a function which returns a training batch (batch_x, batch_y) for
                                  a given list of ids

        Keyword Arguments:
            shuffle {bool} -- If the ids should be shuffled (default: {True})
            epoch_len {int} -- Number of samples per epoch (not number of batches). If `None` the
                               number of ids is used. (default: {None})
        """
        self.ids = ids
        self.batch_size = batch_size

        self.data_fn = data_fn

        self.shuffle = shuffle
        # NOTE: epoch_len is the number of samples per epoch not number of batches
        if epoch_len is not None:
            self.epoch_len = epoch_len
        else:
            self.epoch_len = len(ids)
        # Round epoch length down according to the batch size
        self.epoch_len = batch_size * (self.epoch_len // batch_size)
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.epoch_len // self.batch_size

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        # Prepare the indexes for this epoch
        num_samples = len(self.ids)
        epoch_len = self.epoch_len
        if num_samples <= epoch_len:
            # Less samples than needed per epoch
            all_indexes = []
            # Repeat all indices
            for _ in range(math.ceil(epoch_len / num_samples)):
                all_indexes.extend(self._create_indexes_array())
            self.indexes = all_indexes
        else:
            # More samples than needed per epoch
            if hasattr(self, 'indexes') and len(self.indexes) >= epoch_len * 2:
                # Enough samples left for one mor epoch
                self.indexes = self.indexes[epoch_len:]
            else:
                # Not enough samples left
                self.indexes = self._create_indexes_array()

    def _create_indexes_array(self):
        indexes = np.arange(len(self.ids))
        if self.shuffle is True:
            np.random.shuffle(indexes)
        return indexes

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Indices of this batch
        idxs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        ids = [self.ids[k] for k in idxs]

        # Apply data augmentation
        batch_x, batch_y = self.data_fn(ids)

        return batch_x, batch_y


def data_generator_from_lists(batch_size, data_x, data_y, dataaug_fn, prepare_fn,
                              normalize_fn=None, **kwargs):
    ids = list(range(len(data_x)))
    data_fn = data_fn_from_lists(data_x, data_y, dataaug_fn, prepare_fn, normalize_fn)
    return DataGenerator(ids, batch_size, data_fn, **kwargs)


def data_generator_for_validation(val_x, val_y, prepare_fn, normalize_fn=None):
    ids = list(range(len(val_x)))
    data_fn = data_fn_from_lists(val_x, val_y, lambda x: x, prepare_fn, normalize_fn)
    return DataGenerator(ids, 1, data_fn, shuffle=False)


def data_fn_from_lists(data_x, data_y, dataaug_fn, prepare_fn, normalize_fn=None):
    def data_fn(ids):
        # Loading
        batch_x = [data_x[i] for i in ids]
        batch_y = [data_y[i] for i in ids]

        # Normalize if a normalizer is given
        if normalize_fn is not None:
            batch_x = [normalize_fn(x) for x in batch_x]

        # Dataaug
        batch_x, batch_y = dataaug_fn([batch_x, batch_y])

        # Prepare for training
        return prepare_fn(batch_x, batch_y)
    return data_fn
