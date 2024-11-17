# %%
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

import easyfigs.basicplot as bp
from config_global import EXP_DIR

# %%
record_layers = ['layer1.0.relu', 'layer2.0.relu', 'layer3.0.relu', 'layer4.0.relu', 'avgpool', 'fc']
def read_results(start_id, end_id, exp_name, decode_target):
    return pd.concat([pd.read_csv(Path(EXP_DIR).joinpath(exp_name, f'run_{run_id:04d}', f'{decode_target}_decoding_results_240820.csv'), index_col=0) for run_id in range(start_id, end_id)])

# %%
def cat_decoding_plot(full_data_r, one_cat_data_r, title, suffix):
    x_axis = np.arange(len(record_layers))
    x_offset = -0.1
    u1, p_value = mannwhitneyu(np.array(full_data_r), np.array(one_cat_data_r))
    indicator = p_value < 0.05
    
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.errorbar(x_axis, full_data_r.mean(), yerr=full_data_r.std(), fmt='o-', capsize=3, label='Full cat. var.', color='C1', alpha=0.8)
    ax.errorbar(x_axis + x_offset, one_cat_data_r.mean(), yerr=one_cat_data_r.std(), fmt='o-', capsize=3, label='Reduced cat. var.', color='grey')
    ax.axhline(y=1.0/117, color='grey', linestyle='--', label='Chance')
    for i, ind in enumerate(indicator):
        if ind:
            ax.text(x_axis[i], 0.0, '*', color='k', ha='center', va='center', fontsize=12)

    ax.set_xticks(x_axis, record_layers, rotation=-20)
    ax.set_yticks([0, 0.05, 0.1])
    ax.set_ylabel('Category decode accuracy')
    ax.set_xlabel('Decoding layer')
    ax.set_title(title)
    ax.set_ylim([0, None])
    ax.legend(loc='best', fontsize='small')
    bp.remove_top_right_spines(ax)
    fig.tight_layout()
    fig.savefig(f'figures/ctrl_cat_var_240927_cat_decoding_model_target_{suffix}.pdf', transparent=True, bbox_inches='tight')

read_cat_results = partial(read_results, exp_name='ctrl_cat_var_240927', decode_target='cat')
cat_decoding_plot(read_cat_results(0, 6), read_cat_results(24, 30), 'Model target: Distance', 'dis')
cat_decoding_plot(read_cat_results(6, 12), read_cat_results(30, 36), 'Model target: Translation', 'tran')
cat_decoding_plot(read_cat_results(12, 18), read_cat_results(36, 42), 'Model target: Rotation', 'rot')
cat_decoding_plot(read_cat_results(18, 24), read_cat_results(42, 48), 'Model target: Dis. Tran. Rot.', 'dis_tran_rot')

# %%
cat_model_r = read_results(18, 24, 'ctrl_trans_var_240927', 'cat') # cateogry trained models
untrained_model_r =read_results(0, 5, 'pretrain_and_random_resnet18_0220', 'cat') # random models
# read pixel decoding results
pixel_decode_r = np.array(pd.read_csv('experiments/pixel_decoding/241116_pixel_decoding_results.csv', index_col=0)).reshape(-1)
pixel_mean, pixel_std = np.mean(pixel_decode_r), np.std(pixel_decode_r, ddof=1)

def cat_decoding_plot_compare(full_data_r, one_cat_data_r, target, suffix):
    x_axis = np.arange(len(record_layers))
    x_offset = -0.1
    
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.errorbar(-1, pixel_mean, pixel_std, fmt='o-', capsize=3, label='Pixels decoding', color='C4', alpha=0.8)
    
    ax.errorbar(x_axis - x_offset, cat_model_r.mean(), yerr=cat_model_r.std(), fmt='o-', capsize=3, label='Obj. category (full cat. var.)', color='C0', alpha=0.8)
    ax.errorbar(x_axis, full_data_r.mean(), yerr=full_data_r.std(), fmt='o-', capsize=3, label=f'{target} (full cat. var.)', color='C1', alpha=0.8)
    ax.errorbar(x_axis + x_offset, one_cat_data_r.mean(), yerr=one_cat_data_r.std(), fmt='o-', capsize=3, label=f'{target} (reduced cat. var.)', color='grey')
    ax.errorbar(x_axis + 2 * x_offset, untrained_model_r.mean(), yerr=untrained_model_r.std(), fmt='o-', capsize=3, label='Untrained', color='#55486F')

    ax.axhline(y=1.0/117, color='grey', linestyle='--', label='Chance')
    ax.set_xticks(np.arange(-1, len(record_layers)), ['pixels'] + record_layers, rotation=-40)
    ax.set_ylabel('Category decode accuracy \n (log scale)')
    ax.set_xlabel('Decoding layer')
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize='xx-small')
    bp.remove_top_right_spines(ax)
    fig.tight_layout()
    fig.savefig(f'figures/ctrl_cat_var_240927_cat_decoding_model_target_{suffix}_compare.pdf', transparent=True, bbox_inches='tight')

read_cat_results = partial(read_results, exp_name='ctrl_cat_var_240927', decode_target='cat')
cat_decoding_plot_compare(read_cat_results(0, 6), read_cat_results(24, 30), 'Distance', 'dis')
cat_decoding_plot_compare(read_cat_results(6, 12), read_cat_results(30, 36), 'Translation', 'tran')
cat_decoding_plot_compare(read_cat_results(12, 18), read_cat_results(36, 42), 'Rotation', 'rot')
cat_decoding_plot_compare(read_cat_results(18, 24), read_cat_results(42, 48), 'Dis. Tran. Rot.', 'dis_tran_rot')

# %%
def x_decoding_plot(full_data_r, no_tran_data_r, title, suffix):
    x_axis = np.arange(len(record_layers))
    x_offset = -0.1
    u1, p_value = mannwhitneyu(np.array(full_data_r), np.array(no_tran_data_r))
    indicator = p_value < 0.05
    
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.errorbar(x_axis, full_data_r.mean(), yerr=full_data_r.std(), fmt='o-', capsize=3, label='Full tran. var.', color='C1', alpha=0.8)
    ax.errorbar(x_axis + x_offset, no_tran_data_r.mean(), yerr=no_tran_data_r.std(), fmt='o-', capsize=3, label='Reduced tran. var.', color='grey')    
    ax.set_xticks(x_axis, record_layers, rotation=-20)
    ax.set_ylabel('X decode performance')
    ax.set_xlabel('Decoding layer')
    ax.set_title(title)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.legend(loc=(0.25, 0.08), fontsize='small')

    ylim = ax.get_ylim()
    for i, ind in enumerate(indicator):
        if ind:
            ax.text(x_axis[i], ylim[0], '*', color='k', ha='center', va='center', fontsize=12)

    bp.remove_top_right_spines(ax)
    fig.tight_layout()
    fig.savefig(f'figures/ctrl_trans_var_240927_x_decoding_model_target_{suffix}.pdf', transparent=True, bbox_inches='tight')

read_x_results = partial(read_results, exp_name='ctrl_trans_var_240927', decode_target='x')
x_decoding_plot(read_x_results(0, 6), read_x_results(24, 30), 'Model target: Distance', 'dis')
x_decoding_plot(read_x_results(6, 12), read_x_results(30, 36), 'Model target: Rotation', 'rot')
x_decoding_plot(read_x_results(12, 18), read_x_results(36, 42), 'Model target: Dis. Rot.', 'dis_rot')
x_decoding_plot(read_x_results(18, 24), read_x_results(42, 48), 'Model target: Obj. category', 'obj_cat')

# %%
def y_decoding_plot(full_data_r, no_tran_data_r, title, suffix):
    x_axis = np.arange(len(record_layers))
    x_offset = -0.1
    u1, p_value = mannwhitneyu(np.array(full_data_r), np.array(no_tran_data_r))
    indicator = p_value < 0.05
    
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.errorbar(x_axis, full_data_r.mean(), yerr=full_data_r.std(), fmt='o-', capsize=3, label='Full tran. var.', color='C1', alpha=0.8)
    ax.errorbar(x_axis + x_offset, no_tran_data_r.mean(), yerr=no_tran_data_r.std(), fmt='o-', capsize=3, label='Reduced tran. var.', color='grey')    
    ax.set_xticks(x_axis, record_layers, rotation=-20)
    ax.set_ylabel('Y decode performance')
    ax.set_xlabel('Decoding layer')
    ax.set_title(title)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.legend(loc=(0.25, 0.08), fontsize='small')

    ylim = ax.get_ylim()
    for i, ind in enumerate(indicator):
        if ind:
            ax.text(x_axis[i], ylim[0], '*', color='k', ha='center', va='center', fontsize=12)

    bp.remove_top_right_spines(ax)
    fig.tight_layout()
    fig.savefig(f'figures/ctrl_trans_var_240927_y_decoding_model_target_{suffix}.pdf', transparent=True, bbox_inches='tight')

read_y_results = partial(read_results, exp_name='ctrl_trans_var_240927', decode_target='y')
y_decoding_plot(read_y_results(0, 6), read_y_results(24, 30), 'Model target: Distance', 'dis')
y_decoding_plot(read_y_results(6, 12), read_y_results(30, 36), 'Model target: Rotation', 'rot')
y_decoding_plot(read_y_results(12, 18), read_y_results(36, 42), 'Model target: Dis. Rot.', 'dis_rot')
y_decoding_plot(read_y_results(18, 24), read_y_results(42, 48), 'Model target: Obj. category', 'obj_cat')

# %%



