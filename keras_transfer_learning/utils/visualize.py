"""Tools for visualizing results.
"""
import os
import glob
import re

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.ndimage import filters

NAME_PARTS = [
    'Experiment',
    'Backbone',
    'Head',
    'Dataset',
    'Initialization',
    'Size'
]


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


def get_models(pattern: str, model_dirs: list = None):
    # Default model dirs
    if model_dirs is None:
        model_dirs = sorted(
            [f for f in glob.glob(os.path.join('.', 'models', '*'))])

    prog = re.compile(pattern)
    selected_model_dirs = [
        m for m in model_dirs if prog.match(m.split(os.path.sep)[-1])]
    if selected_model_dirs == []:
        raise ValueError(
            'Could not find a model for the patter {}.'.format(pattern))
    return selected_model_dirs


def plot_over_epoch(pattern: str, metric :str, model_dirs: list = None, size: tuple = None):
    selected_models = get_models(pattern, model_dirs)
    return _plot_map_over_epoch_df(_get_results_df(selected_models, metric), size)


def plot_last(pattern: str, metric: str, model_dirs: list = None, size: tuple = None,
                  ignore_experiment=False, ignore_backbone=False, ignore_head=False,
                  ignore_dataset=False, ignore_init=False, ignore_size=False):
    ignored_vals = []
    if ignore_experiment:
        ignored_vals.append('Experiment')
    if ignore_backbone:
        ignored_vals.append('Backbone')
    if ignore_head:
        ignored_vals.append('Head')
    if ignore_dataset:
        ignored_vals.append('Dataset')
    if ignore_init:
        ignored_vals.append('Initialization')
    if ignore_size:
        ignored_vals.append('Size')
    selected_model_dirs = get_models(pattern, model_dirs)
    results_last = _get_results_last(selected_model_dirs, metric)
    return _plot_last(results_last, size, ignored_vals)


def _split_model_name(model_name):
    vals = dict(zip(NAME_PARTS, model_name.split('_')))
    vals['Initialization'] = 'pretrained' if vals['Initialization'] == 'P' else 'random'
    vals['Size'] = 'F' if vals['Size'] == 'F' else int(vals['Size'])
    return vals


def _get_model_name(path):
    return path.rpartition(os.path.sep)[-1]


def _get_model_results(path):
    results_file = os.path.join(path, 'results.csv')
    return pd.read_csv(results_file)


def _get_results_df(dirs, metric: str):
    results = {_get_model_name(p): _get_model_results(p) for p in dirs}
    results_metric = {name: df[metric] for name, df in results.items()}
    return pd.DataFrame(results_metric)


def _get_results_last(dirs, metric: str):
    results = {_get_model_name(p): _get_model_results(p) for p in dirs}
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


def _plot_last(results_last, size: tuple = None, ignored_vals: list = None):
    if ignored_vals is None:
        ignored_vals = []

    def _create_datapoint(n, v):
        desc = _split_model_name(n)
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

    # Draw the plot
    fig = _create_figure(size)
    ax = sns.barplot(data=df, y='mAP', x=x_label, hue=hue_label)

    # Set ylim based on values
    min_map = df['mAP'].min()
    max_map = df['mAP'].max()
    buffer = (max_map - min_map) * 0.2
    ax.set(ylim=(min_map - buffer, max_map + buffer))
    return fig
