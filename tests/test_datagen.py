import pytest

import numpy as np

from keras_transfer_learning.data import datagen


def _default_data_fn(ids):
    return ids, None


def _create_datagen(ids=range(101), batch_size=8, data_fn=_default_data_fn, shuffle=False,
                    epoch_len=None):
    return datagen.DataGenerator(ids, batch_size, data_fn, shuffle, epoch_len)


def test_data_fn_input():
    """Tests if the data function of a generator gets the expected input"""
    batch_size = 8

    def data_fn(ids):
        assert isinstance(ids, list), \
            'Data functions are expecting lists but got {}'.format(type(ids))
        assert len(ids) == batch_size, \
            'Length of ids is expected to be the batch size {} but was {}'.format(
                batch_size, len(ids))
        assert ids[0] == batch_id * batch_size, \
            'First id expeced to be {} in batch {} but was {}'.format(
                batch_id * batch_size, batch_id, ids[0])
        return None, None

    gen = _create_datagen(ids=range(100), batch_size=batch_size,
                          data_fn=data_fn, shuffle=False)

    batch_id = 0
    _ = gen[batch_id]
    batch_id = 1
    _ = gen[batch_id]


def test_len():
    # Only one batch
    gen = _create_datagen(ids=range(8), batch_size=8)
    assert len(gen) == 1

    # 9 batches
    gen = _create_datagen(ids=range(90), batch_size=10)
    assert len(gen) == 9

    # Batch size doesn't fit len(ids)
    gen = _create_datagen(ids=range(20), batch_size=6)
    assert len(gen) == 3

    # Epoch len fixed
    # Only one batch
    gen = _create_datagen(ids=range(50), batch_size=8, epoch_len=8)

    # 9 batches
    gen = _create_datagen(ids=range(50), batch_size=10, epoch_len=90)
    assert len(gen) == 9

    # Batch size doesn't fit len(ids)
    gen = _create_datagen(ids=range(50), batch_size=6, epoch_len=20)
    assert len(gen) == 3


def test_smaller_number_of_samples():
    num_samples = 9
    batch_size = 8
    epoch_len = 20

    gen = _create_datagen(ids=range(num_samples), batch_size=batch_size,
                          shuffle=False, epoch_len=epoch_len)
    assert len(gen) == 2

    ids, _ = gen[0]
    assert len(ids) == batch_size
    assert ids == list(range(8))

    ids, _ = gen[1]
    assert len(ids) == batch_size
    assert ids == [8] + list(range(7))


def test_larger_number_of_samples():
    num_samples = 34
    batch_size = 8
    epoch_len = 20

    gen = _create_datagen(ids=range(num_samples), batch_size=batch_size,
                          shuffle=False, epoch_len=epoch_len)
    assert len(gen) == 2

    ids, _ = gen[0]
    assert len(ids) == batch_size
    assert ids == list(range(8))

    ids, _ = gen[1]
    assert len(ids) == batch_size
    assert ids == list(range(8, 16))

    gen.on_epoch_end()
    assert len(gen) == 2

    ids, _ = gen[0]
    assert len(ids) == batch_size
    assert ids == list(range(16, 24))

    ids, _ = gen[1]
    assert len(ids) == batch_size
    assert ids == list(range(24, 32))

    gen.on_epoch_end()
    assert len(gen) == 2

    ids, _ = gen[0]
    assert len(ids) == batch_size
    assert ids == list(range(8))

    ids, _ = gen[1]
    assert len(ids) == batch_size
    assert ids == list(range(8, 16))


def test_shuffle():
    # Seed the numpy random number generator
    np.random.seed(1)

    gen = _create_datagen(ids=range(100), batch_size=50, shuffle=True)

    assert len(gen) == 2
    batch00, _ = gen[0]
    batch01, _ = gen[1]

    gen.on_epoch_end()
    assert len(gen) == 2

    batch10, _ = gen[0]
    batch11, _ = gen[1]

    assert batch00 != batch01
    assert batch10 != batch11

    assert batch00 != batch10
    assert batch01 != batch11

    assert np.sort(batch00 + batch01).tolist() == np.sort(batch10 + batch11).tolist()
