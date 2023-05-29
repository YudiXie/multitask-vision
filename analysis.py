import os

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
             ):
    """
        make a bar plot of the performance from a config_df
        that has two varying parameters and different random seeds.
        each first variable parameter dimension is a group of bars with the same color
        second variable parameters are different bars on the x axis

        args:
            data: dict, the data to plot, keys are the group names, values are the data
            x_axis_labels: list, the labels for the x axis
            group_names: list, a list of strings that are the names of the groups 
                each group has many bars with the same color, 
                shown from left to right in the order of the groups
            data_err: dict, the error bar data, keys are the group names,
                not all keys in the data is required to have error bar data
                values are the error bar data
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
    plt.close()


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(EXP_DIR, 'multi_task_vs_categorization0527', 'mt0527_resnet18.csv'), index_col=0)

    group_names = ['Pre-trained', 'Categorization', 'Multi-task']
    x_axis_labels = ['V1', 'V2', 'V4', 'IT', 'Behavior']
    benchmark_list = [
        'movshon.FreemanZiemba2013public.V1-pls',
        'movshon.FreemanZiemba2013public.V2-pls',
        'dicarlo.MajajHong2015public.V4-pls',
        'dicarlo.MajajHong2015public.IT-pls',
        'dicarlo.Rajalingham2018public-i2n',
        ]
    data_dict = {}
    error_dict = {}

    for group in group_names:
        data_dict.update({group: list(df[df['exp_group'] == group].groupby('benchmark').mean()['score'].reindex(benchmark_list))})
        
    for group in group_names[1:]:
        error_dict.update({group: list(df[df['exp_group'] == group].groupby('benchmark').std()['score'].reindex(benchmark_list))})

    bar_2par(data_dict, x_axis_labels, group_names, error_dict,
            exp_name='multi_task_vs_categorization0527', 
            fig_name='compare_different_groups',
            y_label='Score',)
    