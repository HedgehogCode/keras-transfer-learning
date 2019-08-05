"""Tools for visualizing results.
"""
import os
import re
import math

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.ndimage import filters

from keras_transfer_learning import utils
from keras_transfer_learning.utils.utils import INIT_NAME, PRE_DATA_NAME, DATA_NAME, HEAD_NAME
from keras_transfer_learning.utils.utils import BACKBONE_NAME, NUM_TRAIN_NAME, RUN_NAME, NAME_PARTS


def set_default_plotting():
    small_size = 8
    medium_size = 10
    large_size = 12
    sns.set(rc={'font.size': small_size,
                'axes.titlesize': small_size,
                'axes.labelsize': medium_size,
                'xtick.labelsize': small_size,
                'ytick.labelsize': small_size,
                'legend.fontsize': small_size,
                'figure.titlesize': large_size})


def get_models(pattern: str):
    model_names = utils.utils.list_model_names(pattern=pattern)

    if model_names == []:
        raise ValueError(
            'Could not find a model for the patter {}.'.format(pattern))
    return model_names


def plot_over_epoch(pattern: str, metric: str, size: tuple = None):
    selected_models = get_models(pattern)
    return _plot_map_over_epoch_df(_get_results_df(selected_models, metric), size)


def plot_last(pattern: str, metric: str, size: tuple = None,
              ignore_run=False, ignore_backbone=False, ignore_head=False,
              ignore_dataset=False, ignore_init=False, ignore_num_train=False,
              ignore_pre_data=False, plot_type='barplot'):
    ignored_vals = []
    if ignore_run:
        ignored_vals.append(RUN_NAME)
    if ignore_backbone:
        ignored_vals.append(BACKBONE_NAME)
    if ignore_head:
        ignored_vals.append(HEAD_NAME)
    if ignore_dataset:
        ignored_vals.append(DATA_NAME)
    if ignore_init:
        ignored_vals.append(INIT_NAME)
    if ignore_num_train:
        ignored_vals.append(NUM_TRAIN_NAME)
    if ignore_pre_data:
        ignored_vals.append(PRE_DATA_NAME)
    selected_models = get_models(pattern)
    results_last = _get_results_last(selected_models, metric)
    return _plot_last(results_last, size, ignored_vals, plot_type)


def _get_model_results(name):
    results_file = os.path.join('models', name, 'results.csv')
    return pd.read_csv(results_file)


def _get_results_df(names, metric: str):
    results = {name: _get_model_results(name) for name in names}
    results_metric = {name: df[metric] for name, df in results.items()}
    return pd.DataFrame(results_metric)


def _get_results_last(names, metric: str):
    results = {name: _get_model_results(name) for name in names}
    return {n: df[metric].iloc[-1] for n, df in results.items()}


def _create_figure(size: tuple = None):
    if size is None:
        size = (6.4, 4.8)
    return plt.figure(figsize=size, dpi=300)


def _plot_map_over_epoch_df(df, size: tuple = None):

    def smooth_dataset_gaussian(data, sigma=1.5):
        return data.apply(lambda x: filters.gaussian_filter(x, sigma, mode='nearest'))

    def smooth_dataset_box(data, width):
        box = np.ones(width) / width
        return data.apply(lambda x: filters.convolve(x, box, mode='nearest'))

    fig = _create_figure(size)
    sns.lineplot(data=smooth_dataset_gaussian(df))
    return fig


def _plot_last(results_last, size: tuple = None, ignored_vals: list = None,
               plot_type: str = 'barplot'):
    if ignored_vals is None:
        ignored_vals = []

    def _create_datapoint(n, v):
        desc = utils.utils.split_model_name(n)
        desc['mAP'] = v
        return desc

    datapoints = [_create_datapoint(n, v) for n, v in results_last.items()]
    df = pd.DataFrame(datapoints)

    # Find x and hue
    different_vals = {n: len(df[n].unique())
                      for n in NAME_PARTS if n not in ignored_vals}
    x_label, hue_label = sorted(
        different_vals, key=different_vals.get, reverse=True)[:2]
    if different_vals[hue_label] == 1:
        hue_label = None

    # Find the number of observations
    if hue_label is not None:
        nobs = df.groupby([x_label, hue_label])['mAP'].agg(['count'])
    else:
        nobs = df.groupby([x_label])['mAP'].agg(['count'])
    nobs = np.reshape(nobs.values, [-1])

    # Draw the plot
    fig = _create_figure(size)
    if plot_type == 'barplot':
        ax = sns.barplot(data=df, y='mAP', x=x_label, hue=hue_label)
    elif plot_type == 'boxplot':
        ax = sns.boxenplot(data=df, y='mAP', x=x_label, hue=hue_label)

    # Set ylim based on values
    min_map = df['mAP'].min()
    max_map = df['mAP'].max()
    buffer = (max_map - min_map) * 0.2
    ax.set(ylim=(min_map - buffer, max_map + buffer))

    # Add the number of observations
    if plot_type == 'barplot':
        x_pos = lambda a: a.get_x() + 0.5 * a.get_width()
        y_pos = lambda a: a.get_height() - 0.3 * buffer
        color = 'w'

        positions = sorted([(x_pos(a), y_pos(a))
                            for a in ax.patches if not math.isnan(a.get_height())])
        for (pos_x, pos_y), n in zip(positions, nobs):
            if n > 1:
                ax.text(pos_x, pos_y, 'n:' + str(n),
                        ha='center', size='x-small', color=color)
    return fig
