"""Tools for visualizing results.
"""
import os
import math

import seaborn as sns
from seaborn.categorical import _CategoricalPlotter
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import pandas as pd
from scipy.ndimage import filters

from keras_transfer_learning import utils
from keras_transfer_learning.utils.utils import INIT_NAME, PRE_DATA_NAME, DATA_NAME, HEAD_NAME
from keras_transfer_learning.utils.utils import BACKBONE_NAME, NUM_TRAIN_NAME, RUN_NAME, NAME_PARTS

DEFAULT_FIGURE_SIZE = (6.4, 4.8)
DEFAULT_FIGURE_SIZE_TEX = (2.1, 1.6)


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


def set_default_plotting_tex():
    scaling = 0.5
    base_context = {
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,

        "axes.linewidth": 1.25,
        "grid.linewidth": 1,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "patch.linewidth": 1,

        "xtick.major.width": 1.25,
        "ytick.major.width": 1.25,
        "xtick.minor.width": 1,
        "ytick.minor.width": 1,

        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 4,
        "ytick.minor.size": 4,
    }
    context_dict = {k: v * scaling for k, v in base_context.items()}
    sns.set(font='serif')
    sns.set_style("ticks", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })
    sns.set_context(context_dict)


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
    results_last_df = _get_results_last_df(selected_models, metric)
    return _plot_last(results_last_df, size, ignored_vals, plot_type)


def plot_history(pattern: str, metric: str, hue: str, style: str, size: tuple = None):
    selected_models = get_models(pattern)
    history_df = _get_model_histories_df(selected_models)
    fig = _create_figure(size)
    ax = sns.lineplot(data=history_df, x='Step',
                      y=metric, hue=hue, style=style)
    # ax.set(ylim=(0.06, 0.15))
    # Cheating to make it look more impressive!
    # They did it in the U-Net Fiji Plugin paper
    # ax.set(xscale='log')
    return fig


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


def _get_results_last_df(names, metric: str = None):
    results = {name: _get_model_results(name) for name in names}
    results_last = {name: df.iloc[-1] for name, df in results.items()}

    def _create_datapoint(n, v):
        desc = utils.utils.split_model_name(n)
        desc.update(v.to_dict())
        if metric is not None:
            desc['mAP'] = desc[metric]
        return desc

    datapoints = [_create_datapoint(n, v) for n, v in results_last.items()]
    return pd.DataFrame(datapoints)


def _get_model_history_df(name):
    history_file = os.path.join('models', name, 'history.csv')
    history_df = pd.read_csv(history_file)
    history_df = history_df.rename(columns={'Unnamed: 0': 'Step'})
    model_desc = utils.utils.split_model_name(name)
    for k, v in model_desc.items():
        history_df[k] = v
    return history_df


def _get_model_histories_df(names):
    history_df: pd.DataFrame = None
    for name in names:
        current_df = _get_model_history_df(name)
        if history_df is None:
            history_df = current_df
        else:
            history_df = history_df.append(current_df)
    return history_df.reset_index()


def _create_figure(size: tuple = None):
    if size is None:
        size = DEFAULT_FIGURE_SIZE
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


def _plot_last(results_last_df, size: tuple = None, ignored_vals: list = None,
               plot_type: str = 'barplot'):
    if ignored_vals is None:
        ignored_vals = []

    # Find x and hue
    different_vals = {n: len(results_last_df[n].unique())
                      for n in NAME_PARTS if n not in ignored_vals}
    x_label, hue_label = sorted(
        different_vals, key=different_vals.get, reverse=True)[:2]
    if different_vals[hue_label] == 1:
        hue_label = None

    # Find the number of observations
    if hue_label is not None:
        nobs = results_last_df.groupby([x_label, hue_label])[
            'mAP'].agg(['count'])
    else:
        nobs = results_last_df.groupby([x_label])['mAP'].agg(['count'])
    nobs = np.reshape(nobs.values, [-1])

    # Draw the plot
    fig = _create_figure(size)
    if plot_type == 'barplot':
        ax = sns.barplot(data=results_last_df, y='mAP',
                         x=x_label, hue=hue_label)
    elif plot_type == 'boxplot':
        ax = sns.boxenplot(data=results_last_df, y='mAP',
                           x=x_label, hue=hue_label)
    elif plot_type == 'swarmplot':
        ax = sns.swarmplot(data=results_last_df, y='mAP',
                           x=x_label, hue=hue_label, dodge=True)
    else:
        raise ValueError(f'Unknown plot type: {plot_type}')

    # Set ylim based on values
    min_map = results_last_df['mAP'].min()
    max_map = results_last_df['mAP'].max()
    buffer = (max_map - min_map) * 0.2
    ax.set(ylim=(min_map - buffer, max_map + buffer))

    # Add the number of observations
    if plot_type == 'barplot':
        def x_pos(a): return a.get_x() + 0.5 * a.get_width()
        def y_pos(a): return a.get_height() - 0.3 * buffer
        color = 'w'

        positions = sorted([(x_pos(a), y_pos(a))
                            for a in ax.patches if not math.isnan(a.get_height())])
        for (pos_x, pos_y), n in zip(positions, nobs):
            if n > 1:
                ax.text(pos_x, pos_y, 'n:' + str(n),
                        ha='center', size='x-small', color=color)
    return fig


def save(fig, file):
    fig.savefig(file, bbox_inches='tight')


# Copy of seaborn.catplot (BSD-3-Clause) withou the deprectaion handling
# And with a fixed plot type
def catplot(x=None, y=None, hue=None, data=None, row=None, col=None,
            col_wrap=None, order=None, hue_order=None, row_order=None,
            col_order=None, height=5, aspect=1,
            orient=None, color=None, palette=None,
            legend=True, legend_out=True, sharex=True, sharey=True,
            margin_titles=False, facet_kws=None, **kwargs):

    # Determine the plotting function
    plot_func = sns.swarmplot

    # Alias the input variables to determine categorical order and palette
    # correctly in the case of a count plot
    x_, y_ = x, y

    # Determine the order for the whole dataset, which will be used in all
    # facets to ensure representation of all data in the final plot
    p = _CategoricalPlotter()
    p.establish_variables(x_, y_, hue, data, orient, order, hue_order)
    order = p.group_names
    hue_order = p.hue_names

    # Determine the palette to use
    # (FacetGrid will pass a value for ``color`` to the plotting function
    # so we need to define ``palette`` to get default behavior for the
    # categorical functions
    p.establish_colors(color, palette, 1)

    # Determine keyword arguments for the facets
    facet_kws = {} if facet_kws is None else facet_kws
    facet_kws.update(
        data=data, row=row, col=col,
        row_order=row_order, col_order=col_order,
        col_wrap=col_wrap, height=height, aspect=aspect,
        sharex=sharex, sharey=sharey,
        legend_out=legend_out, margin_titles=margin_titles,
        dropna=False,
    )

    # Determine keyword arguments for the plotting function
    plot_kws = dict(
        order=order, hue_order=hue_order,
        orient=orient, color=color, palette=palette,
    )
    plot_kws.update(kwargs)

    # Initialize the facets
    g = sns.FacetGrid(**facet_kws)

    # Draw the plot onto the facets
    g.map_dataframe(plot_func, x, y, hue, **plot_kws)

    # Special case axis labels for a count type plot
    if legend and (hue is not None) and (hue not in [x, row, col]):
        g.add_legend(title=hue, label_order=hue_order)

    return g
