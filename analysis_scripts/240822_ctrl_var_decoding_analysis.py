# %%
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.errorbar(x_axis + x_offset, full_data_r.mean(), yerr=full_data_r.std(), fmt='o-', capsize=3, label='Full cat. var.', color='C1', alpha=0.8)
    ax.errorbar(x_axis, one_cat_data_r.mean(), yerr=one_cat_data_r.std(), fmt='o-', capsize=3, label='Reduced cat. var.', color='grey')
    ax.axhline(y=1.0/117, color='grey', linestyle='--', label='Chance')
    ax.set_xticks(x_axis, record_layers, rotation=-20)
    ax.set_ylabel('Category decode accuracy')
    ax.set_xlabel('Decoding layer')
    ax.set_title(title)
    ax.set_ylim([0, None])
    ax.legend(loc='best', fontsize='small')
    bp.remove_top_right_spines(ax)
    fig.tight_layout()
    fig.savefig(f'figures/ctrl_var_target_dist_240712_cat_decoding_model_target_{suffix}.pdf', transparent=True)

read_cat_results = partial(read_results, exp_name='ctrl_var_target_dist_240712', decode_target='cat')
cat_decoding_plot(read_cat_results(0, 3), read_cat_results(12, 15), 'Target: Distance', 'dis')
cat_decoding_plot(read_cat_results(3, 6), read_cat_results(15, 18), 'Target: Translation', 'tran')
cat_decoding_plot(read_cat_results(6, 9), read_cat_results(18, 21), 'Target: Rotation', 'rot')
cat_decoding_plot(read_cat_results(9, 12), read_cat_results(21, 24), 'Target: Dis. Tran. Rot.', 'dis_tran_rot')

# %%
def x_decoding_plot(full_data_r, no_tran_data_r, title, suffix):
    x_axis = np.arange(len(record_layers))
    x_offset = -0.1
    
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.errorbar(x_axis + x_offset, full_data_r.mean(), yerr=full_data_r.std(), fmt='o-', capsize=3, label='Full tran. var.', color='C1', alpha=0.8)
    ax.errorbar(x_axis, no_tran_data_r.mean(), yerr=no_tran_data_r.std(), fmt='o-', capsize=3, label='Reduced tran. var.', color='grey')
    ax.set_xticks(x_axis, record_layers, rotation=-20)
    ax.set_ylabel('X decode performance')
    ax.set_xlabel('Decoding layer')
    ax.set_title(title)
    ax.set_ylim([0, None])
    ax.legend(loc='best', fontsize='small')
    bp.remove_top_right_spines(ax)
    fig.tight_layout()
    fig.savefig(f'figures/ctrl_trans_var_240814_x_decoding_model_target_{suffix}.pdf', transparent=True)

read_x_results = partial(read_results, exp_name='ctrl_trans_var_240814', decode_target='x')
x_decoding_plot(read_x_results(0, 5), read_x_results(20, 25), 'Target: Distance', 'dis')
x_decoding_plot(read_x_results(5, 10), read_x_results(25, 30), 'Target: Rotation', 'rot')
x_decoding_plot(read_x_results(10, 15), read_x_results(30, 35), 'Target: Obj. category', 'obj_cat')
x_decoding_plot(read_x_results(15, 20), read_x_results(35, 40), 'Target: Obj. identity', 'obj_id')

# %%
def y_decoding_plot(full_data_r, no_tran_data_r, title, suffix):
    x_axis = np.arange(len(record_layers))
    x_offset = -0.1
    
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.errorbar(x_axis + x_offset, full_data_r.mean(), yerr=full_data_r.std(), fmt='o-', capsize=3, label='Full tran. var.', color='C1', alpha=0.8)
    ax.errorbar(x_axis, no_tran_data_r.mean(), yerr=no_tran_data_r.std(), fmt='o-', capsize=3, label='Reduced tran. var.', color='grey')
    ax.set_xticks(x_axis, record_layers, rotation=-20)
    ax.set_ylabel('Y decode performance')
    ax.set_xlabel('Decoding layer')
    ax.set_title(title)
    ax.set_ylim([0, None])
    ax.legend(loc='best', fontsize='small')
    bp.remove_top_right_spines(ax)
    fig.tight_layout()
    fig.savefig(f'figures/ctrl_trans_var_240814_y_decoding_model_target_{suffix}.pdf', transparent=True)

read_y_results = partial(read_results, exp_name='ctrl_trans_var_240814', decode_target='y')
y_decoding_plot(read_y_results(0, 5), read_y_results(20, 25), 'Target: Distance', 'dis')
y_decoding_plot(read_y_results(5, 10), read_y_results(25, 30), 'Target: Rotation', 'rot')
y_decoding_plot(read_y_results(10, 15), read_y_results(30, 35), 'Target: Obj. category', 'obj_cat')
y_decoding_plot(read_y_results(15, 20), read_y_results(35, 40), 'Target: Obj. identity', 'obj_id')

# %%



