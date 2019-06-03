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


def plot_map_over_epoch(pattern: str, model_dirs: list = None, size: tuple = None):
    selected_models = get_models(pattern, model_dirs)
    if size is None:
        size = (12, 8)
    return _plot_map_over_epoch_df(_get_results_df(selected_models), size)


def plot_map_last_compare(pattern: str, model_dirs: list = None, size: tuple = None):
    def get_train_num(name):
        num = name.split('_')[-1]
        if num == 'F':
            return 'F'
        else:
            return int(num)

    def get_init(name):
        return 'pretrained' if name.split('_')[-2] == 'P' else 'random'

    selected_model_dirs = get_models(pattern, model_dirs)
    if size is None:
        size = (12, 8)
    results_last = _get_results_last(selected_model_dirs)
    df = pd.DataFrame([{
        'name': n,
        'mAP': v,
        'num_train': get_train_num(n),
        'init': get_init(n)
    } for n, v in results_last.items()])

    fig = plt.figure(figsize=size)
    ax = sns.barplot(x='num_train', y='mAP', hue='init', data=df)
    ax.set(ylim=(df.min()['mAP'] - 0.05, df.max()['mAP'] + 0.05), ylabel='mAP')
    return fig


def plot_map_last(pattern: str, model_dirs: list = None, size: tuple=None):
    selected_model_dirs = get_models(pattern, model_dirs)
    if size is None:
        size = (12, 8)
    results_last = _get_results_last(selected_model_dirs)
    df = pd.DataFrame({n: [v] for n, v in results_last.items()})
    return _plot_map_last(df, size)


def _get_model_name(path):
    return path.rpartition(os.path.sep)[-1]


def _get_model_results(path):
    results_file = os.path.join(path, 'results.csv')
    df = pd.read_csv(results_file)
    return df.drop('Unnamed: 0', 1).set_index('epoch')


def _get_results_df(dirs):
    results = {_get_model_name(p): _get_model_results(p) for p in dirs}
    results_mean = {n: df.mean(axis=1) for n, df in results.items()}
    return pd.DataFrame(results_mean)


def _get_results_last(dirs):
    results = {_get_model_name(p): _get_model_results(p) for p in dirs}
    return {n: df.mean(axis=1).iloc[-1] for n, df in results.items()}


def _plot_map_over_epoch_df(df, size):

    def smooth_dataset_gaussian(data, sigma=1.5):
        return data.apply(lambda x: filters.gaussian_filter(x, sigma, mode='nearest'))

    def smooth_dataset_box(data, width):
        box = np.ones(width) / width
        return data.apply(lambda x: filters.convolve(x, box, mode='nearest'))

    fig = plt.figure(figsize=size)
    ax = sns.lineplot(data=smooth_dataset_gaussian(df))
    ax.set(ylim=(0.4, 0.9), ylabel='mAP')
    return fig


def _plot_map_last(df, size):
    fig = plt.figure(figsize=size)
    ax = sns.barplot(data=df)
    # TODO set ylim based on values
    ax.set(ylim=(0.4, 0.9), ylabel='mAP')
    return fig
