import os
import math
from glob import glob

import pandas as pd
import numpy as np


def load_train(data_dir=os.path.join('data', 'pcom'), seed=42, train_val_split=0.9):
    train_dir = os.path.join(data_dir, 'train')
    labels_file = os.path.join(data_dir, 'train_labels.csv')

    df = pd.read_csv(labels_file)
    df = df.sort_values(by=['id'])

    files = sorted(glob(os.path.join(train_dir, '*.tif')))
    df['file'] = files

    # Train/Validation split
    num = len(df)
    num_train = math.floor(num * train_val_split)
    random_idxs = np.random.RandomState(seed).permutation(num)
    train_idxs = random_idxs[:num_train]
    val_idxs = random_idxs[num_train:]

    return df.iloc[train_idxs], df.iloc[val_idxs]
