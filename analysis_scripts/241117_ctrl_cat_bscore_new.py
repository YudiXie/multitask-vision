# %%
import numpy as np
import pandas as pd
from pathlib import Path

from matplotlib import pyplot as plt
from itertools import product

import easyfigs.basicplot as bp
from config_global import EXP_DIR, FIG_DIR

# %%
bscore_df = pd.read_csv(Path(EXP_DIR).joinpath('ctrl_cat_var_240927', 'brainscore_results.csv'), index_col=0)

# %%
dset_list = ['tdw_1m_20240206',
             'tdw_1m_1c_n03001627_20240711']
task_set_list = [
    'distance_reg',
    'translation_reg',
    'rotation_reg',
    'distance_translation_rotation',
]
plot_task_names = ['Distance', 'Translation', 'Rotation', 'Dis. Tran. Rot.']
seed_list = [0, 1, 2, 3, 4, 5]
model_dict = {'model': [], 'dset': [], 'task_set': [], 'seed': []}
for i_s, state in enumerate(product(dset_list, task_set_list, seed_list)):
    dset, task_set, seed = state
    model_dict['model'].append(f'ctrl_cat_var_240927-resnet18-{i_s}-batch--1')
    model_dict['dset'].append(dset)
    model_dict['task_set'].append(task_set)
    model_dict['seed'].append(seed)
model_df = pd.DataFrame(model_dict)

# %%
full_df = pd.merge(bscore_df, model_df, on='model', validate='m:1')
full_neural_df = full_df[full_df['benchmark_region'] != 'Behavior']

# %%
bs_results = full_neural_df.groupby(['dset', 'task_set', 'seed'])['score'].mean().groupby(['dset', 'task_set']).agg(['mean', 'std'])

full_var_r = bs_results.loc[('tdw_1m_20240206', )].reindex(task_set_list)
y1 = list(full_var_r['mean'])
y1_e = list(full_var_r['std'])

red_var_r = bs_results.loc[('tdw_1m_1c_n03001627_20240711', )].reindex(task_set_list)
y2 = list(red_var_r['mean'])
y2_e = list(red_var_r['std'])

# %%
df_pt_rnd = pd.read_csv(Path(EXP_DIR).joinpath('pretrain_and_random_resnet18_0220', 'brainscore_results.csv'), index_col=0)
df_pt_rnd_neural = df_pt_rnd[df_pt_rnd['benchmark_region'] != 'Behavior']
df_pt_rnd_neural_agg = df_pt_rnd_neural.groupby(['exp_group', 'model'])['score'].mean().groupby('exp_group').agg(['mean', 'std'])

rnd_data = df_pt_rnd_neural_agg['mean']['random']
rnd_error = df_pt_rnd_neural_agg['std']['random']

pt_data = df_pt_rnd_neural_agg['mean']['imagenet1k_pretrain']
pt_error = df_pt_rnd_neural_agg['std']['imagenet1k_pretrain']

# %%
plot_data = {
    'Full cat. var.': {'y': y1, 'error': y1_e, 'kwargs': {'color': 'C1', 'alpha': 0.8}},
    'Reduced cat. var.': {'y': y2, 'error': y2_e, 'kwargs': {'color': 'grey'}},
}

fig, ax = plt.subplots(figsize=(3.6, 2.8))
x_axis, _ignore = bp.bar_groups(ax, plot_task_names, plot_data, bar_label=True)

x_s, x_e = ax.get_xlim()
ax.hlines(rnd_data, x_s, x_e, linestyles='dashed', colors='k', label='Untrained')
ax.fill_between([x_s, x_e], 2 * [rnd_data - rnd_error], 2 * [rnd_data + rnd_error], alpha=0.2, color='k')

ax.set_xticks(x_axis, plot_task_names, rotation=-20)
ax.set_ylabel('Mean neural alignment score \n (V1, V2, V4, IT public benchmarks)')
ax.set_xlabel('Target latent')
ax.set_xlim(x_s, x_e)
ax.set_ylim(0.22, 0.4)
ax.legend(loc=(0.48, 0.2), fontsize='small')
bp.remove_top_right_spines(ax)
fig.savefig(Path(FIG_DIR).joinpath('ctrl_cat_var_model_brainscore.pdf'), transparent=True, bbox_inches='tight')


