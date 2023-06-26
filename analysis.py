import os
import scipy.stats

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from config_global import EXP_DIR, FIG_DIR


def adjust_figure(ax=None):
    if ax is None:
        ax = plt.gca()
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    # ax.yaxis.set_ticks_position('left')
    # ax.xaxis.set_ticks_position('bottom')
    plt.tight_layout(pad=0.5)


def get_stat_str(p_value, maxasterix=None):
    """
    Get asterix string based on p value
    :param p_value:
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    :return:
    """
    # * is p < 0.05
    # ** is p < 0.005
    # *** is p < 0.0005
    # etc.
    text = ''
    p = .05

    while p_value < p:
        text += '*'
        p /= 10.

        if maxasterix and len(text) == maxasterix:
            break

    if len(text) == 0:
        text = 'n. s.'

    return text


def barplot_annotate_brackets(idx1, idx2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None):
    """
    Annotate barplot with p-values.
    adapted from:
    https://stackoverflow.com/questions/11517986/indicating-the-statistically-significant-difference-in-bar-graph

    :param idx1: index of left bar to put bracket over
    :param idx2: index of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    """

    if type(data) is str:
        text = data
    else:
        text = get_stat_str(data)

    lx, ly = center[idx1], height[idx1]
    rx, ry = center[idx2], height[idx2]

    if yerr:
        ly += yerr[idx1]
        ry += yerr[idx2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    plt.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)


def two_set_scatter_plot(data1, data2, 
                         labels, title_str, ylabel, 
                         save_str, show=True):
    """
    scatter plot for two groups
        :param data1: pandas series for data group 1
        :param data2: pandas series for data group 1
        :param labels: labels for two groups
        :param title_str: string for title
        :param ylabel: y label string
        :param save_str: string to save
        :param show: whether to show the plot
    """
    mean_data1 = data1.mean()
    sem_data1 = data1.sem()
    data1_x = [1, ] * len(data1)

    mean_data2 = data2.mean()
    sem_data2 = data2.sem()
    data2_x = [2, ] * len(data2)

    # Mann-Whitney U test, or Wilcoxon rank-sum test
    p_value = scipy.stats.mannwhitneyu(data1.dropna(), data2.dropna()).pvalue
    wilcoxon_p_value = scipy.stats.ranksums(data1.dropna(), data2.dropna()).pvalue

    # Shapiro-Wilk test of normality
    l_normal_p = scipy.stats.shapiro(data1.dropna()).pvalue
    r_normal_p = scipy.stats.shapiro(data2.dropna()).pvalue

    # T-test
    ttest_p_value = scipy.stats.ttest_ind(data1.dropna(), data2.dropna()).pvalue

    plt.figure(figsize=(2.5, 5))
    plt.scatter(data1_x, data1, 100, color='w', edgecolors='k', alpha=0.5)
    plt.errorbar([1], mean_data1, yerr=sem_data1,
                 fmt="o",
                 mfc='white',
                 ecolor='k',
                 color='k',
                 elinewidth=1,
                 capsize=10,
                 markersize=10
                 )

    plt.scatter(data2_x, data2, 100, color='mediumorchid', edgecolors='k', alpha=0.5)
    plt.errorbar([2], mean_data2, yerr=sem_data2,
                 fmt="o",
                 mfc='mediumorchid',
                 ecolor='k',
                 color='k',
                 elinewidth=1,
                 capsize=10,
                 markersize=10
                 )

    plt.xticks([1, 2], labels, rotation=45)
    plt.locator_params(nbins=5, axis='y')
    plt.xlim([0.5, 2.5])
    plt.title(title_str
              + "\nMann-Whitney U test P = {:.3f}".format(p_value)
              + "\nWilcoxon rank-sum test P = {:.3f}".format(wilcoxon_p_value)
              + "\nNormality test (left, right) P = ({:.3f}, {:.3f})".format(l_normal_p, r_normal_p)
              + "\nT-test P = {:.3f}".format(ttest_p_value),
              fontsize=12)
    plt.ylabel(ylabel)
    barplot_annotate_brackets(0, 1, p_value, [1, 2],
                              [mean_data1, mean_data2],
                              [data1.max()-mean_data1, data2.max()-mean_data1],
                              barh=0.025)
    adjust_figure()
    plt.savefig(os.path.join('./figures/', save_str + '.pdf'), transparent=True, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def bar_2par(data,
             x_axis_labels,
             group_names,
             data_err={},
             exp_name='exp',
             fig_name='fig',
             legend_title="Groups:",
             x_label=None,
             y_label='Performance',
             fig_title=None,
             ylim=None,
             bar_label=True,
             legend_fontsize=12,
             show=True,
             ):
    """
        make a bar plot of the performance with error bars
        that has two varying parameters
        each first variable parameter dimension is a group of bars with the same color
        second variable parameters are different bars on the x axis

        args:
            data: dict, the data to plot, 
                each key is one elements in group_names
                each value is a list or array that determine the heights of bars
                assume values are all lists of the same length
            x_axis_labels: list, the labels for the x axis
                len(x_axis_labels) is the same as the length of the lists in data
            group_names: list, a list of strings that are the names of the groups
                each group has many bars with the same color,
                this determines the order of colors in the plot
                shown from left to right in the order of the groups
                assume all elements in group_names are keys in data
            data_err: dict, the error bar data, keys are the group names,
                not all keys in the data is required to have error bar data
                values are the error bar data
            fig_name: str, the name of the figure to be saved
    """
    num_groups = len(group_names)
    width = 0.7 / num_groups  # the width of the bars
    x_axis = np.arange(len(x_axis_labels))  # the label locations

    fig, ax = plt.subplots()
    for i, g in enumerate(group_names):
        offset = i * width - ((num_groups - 1) * width / 2)
        kwargs = {}
        if g in data_err:
            kwargs.update({'yerr': data_err[g]})
        
        rect = ax.bar(
            x_axis + offset,
            data[g],
            width,
            label=g,
            capsize=width*15,
            ecolor='black',
            alpha=0.5,
            **kwargs)
        if bar_label:
            ax.bar_label(rect, padding=3, fmt='%.2f', fontsize=8)

    ax.set_xticks(x_axis, x_axis_labels)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend(title=legend_title, fontsize=legend_fontsize)

    if x_label is not None:
        ax.set_xlabel(x_label)

    if y_label is not None:
        ax.set_ylabel(y_label)

    plt.title(fig_title)

    adjust_figure()
    os.makedirs(os.path.join(FIG_DIR, exp_name), exist_ok=True)
    plt.savefig(os.path.join(FIG_DIR, exp_name, fig_name + '.pdf'), transparent=True)
    if show:
        plt.show()
    plt.close()


def scatter_errorbar(data_dict,
                     x_label=None,
                     y_label='Performance',
                     additional_plots=None):
    """
        make a scatter plot with error bars
        args:
            data_dict: dict, the data to plot,
                each key is the group label of a set of points
                each value is a dict with keys 'x', 'y', 'error'
                they are list or array of the same length
                'x' is the x axis data
                'y' is the y axis data
                'error' is the error bar data, which is optional
            additional_plots: list, a list of functions that plot additional 
                things on the same figure, functions will be called without arguements
    """
    for key, value in data_dict.items():
        kwargs = {}
        if 'error' in value:
            kwargs.update({'yerr': value['error']})
        plt.errorbar(value['x'], value['y'], fmt="o", label=key, **kwargs)
    
    if x_label is not None:
        plt.xlabel(x_label)

    if y_label is not None:
        plt.ylabel(y_label)
    
    if additional_plots is not None:
        for plot in additional_plots:
            plot()
    
    adjust_figure()
    plt.legend()
    plt.show()
    plt.close()
