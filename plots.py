import numpy as np
from matplotlib import pyplot as plt


def adjust_figure(ax=None):
    if ax is None:
        ax = plt.gca()
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    # ax.yaxis.set_ticks_position('left')
    # ax.xaxis.set_ticks_position('bottom')
    # plt.tight_layout(pad=0.5)


def data_hist(ax, data, xlabel, ylabel='count', show_mean_std=False):
    ax.hist(data, bins=50)
    if show_mean_std:
        data_mean = np.mean(data)
        data_std = np.std(data)
        ax.axvline(data_mean, color='k', linestyle='dashed')
        ax.axvline(data_mean + data_std, color='r', linestyle='dashed')
        ax.axvline(data_mean - data_std, color='r', linestyle='dashed')
        ax.text(data_mean, 0, f'mean={data_mean:.3f}', rotation=90)
        ax.text(data_mean + data_std, 0, f'std={data_std:.3f}', rotation=90)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    adjust_figure(ax)
