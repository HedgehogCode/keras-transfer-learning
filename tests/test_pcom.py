import math
import pytest

from keras_transfer_learning import data

# @pytest.mark.skip(reason="Needs data")
def test_load_train():
    split = 0.8
    train, test = data.pcom.load_train(train_val_split=split)
    num_train = len(train)
    num_test = len(test)
    num = num_train + num_test

    assert num_train == math.floor(num * split)
