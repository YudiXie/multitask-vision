# %%
import numpy as np
import pandas as pd
from pathlib import Path

from matplotlib import pyplot as plt

from config_global import EXP_DIR, FIG_DIR
from analysis import adjust_figure
from itertools import product

# %%
bscore_df = pd.read_csv(Path(EXP_DIR).joinpath('ctrl_var_target_dist_240712', 'brainscore_results.csv'), index_col=0)

# %%
dset_list = ['tdw_1m_20240206', 
                'tdw_1m_1c_n03001627_20240711']
task_set_list = [
    'distance_reg',
    'translation_reg',
    'rotation_reg',
    'distance_translation_rotation',
]
seed_list = [0, 1, 2]
model_dict = {'model': [], 'dset': [], 'task_set': [], 'seed': []}
for i_s, state in enumerate(product(dset_list, task_set_list, seed_list)):
    dset, task_set, seed = state
    model_dict['model'].append(f'ctrl_var_target_dist_240712-resnet18-{i_s}')
    model_dict['dset'].append(dset)
    model_dict['task_set'].append(task_set)
    model_dict['seed'].append(seed)
model_df = pd.DataFrame(model_dict)

# %%
full_df = pd.merge(bscore_df, model_df, on='model', validate='m:1')
full_neural_df = full_df[full_df['benchmark_region'] != 'Behavior']

# %%
y1 = list(full_neural_df.groupby(['dset', 'task_set', 'seed']).mean(numeric_only=True).groupby(['dset', 'task_set']).mean(numeric_only=True).loc[('tdw_1m_20240206', )].reindex(task_set_list)['score'])
y1_e = list(full_neural_df.groupby(['dset', 'task_set', 'seed']).mean(numeric_only=True).groupby(['dset', 'task_set']).std(numeric_only=True).loc[('tdw_1m_20240206', )].reindex(task_set_list)['score'])

y2 = list(full_neural_df.groupby(['dset', 'task_set', 'seed']).mean(numeric_only=True).groupby(['dset', 'task_set']).mean(numeric_only=True).loc[('tdw_1m_1c_n03001627_20240711', )].reindex(task_set_list)['score'])
y2_e = list(full_neural_df.groupby(['dset', 'task_set', 'seed']).mean(numeric_only=True).groupby(['dset', 'task_set']).std(numeric_only=True).loc[('tdw_1m_1c_n03001627_20240711', )].reindex(task_set_list)['score'])

# %%
df_pt_rnd = pd.read_csv(Path(EXP_DIR).joinpath('pretrain_and_random_resnet50_0220', 'brainscore_results.csv'), index_col=0)
df_pt_rnd_neural = df_pt_rnd[df_pt_rnd['benchmark_region'] != 'Behavior']

rnd_data = df_pt_rnd_neural.groupby(['exp_group', 'model']).mean(numeric_only=True).groupby('exp_group').mean()['score']['random']
rnd_error = df_pt_rnd_neural.groupby(['exp_group', 'model']).mean(numeric_only=True).groupby('exp_group').std(ddof=0)['score']['random']

pt_data = df_pt_rnd_neural.groupby(['exp_group', 'model']).mean(numeric_only=True).groupby('exp_group').mean()['score']['imagenet1k_pretrain']
pt_error = df_pt_rnd_neural.groupby(['exp_group', 'model']).mean(numeric_only=True).groupby('exp_group').std(ddof=0)['score']['imagenet1k_pretrain']

# %%
x = [1, 2, 6, 9]
fig, ax = plt.subplots(figsize=(4.8, 3.6))

ax.errorbar(x, y1, yerr=y1_e, capsize=3, fmt='o', label='tdw_1m_20240206')
ax.errorbar(x, y2, yerr=y2_e, capsize=3, fmt='o', label='tdw_1m_1c_n03001627_20240711')

ax.hlines(pt_data, x[0], x[-1], linestyles='dashed', colors='r', label='ImageNet-1k')
ax.fill_between([x[0], x[-1]], 2 * [pt_data - pt_error], 2 * [pt_data + pt_error], alpha=0.2, color='r')

ax.hlines(rnd_data, x[0], x[-1], linestyles='dashed', colors='k', label='Untrained')
ax.fill_between([x[0], x[-1]], 2 * [rnd_data - rnd_error], 2 * [rnd_data + rnd_error], alpha=0.2, color='k')
ax.legend(loc=(0.05, 0.2))
ax.set_xticks(x)
ax.set_xlabel('Number of output units')
ax.set_ylabel('Mean Brain-Score \n (V1, V2, V4, IT)')
ax.set_yticks([0.25, 0.30, 0.35, 0.40], ['0.25', '', '', '0.40'])
adjust_figure(ax)
fig.tight_layout(pad=0.5)
fig.savefig(Path(FIG_DIR).joinpath('ctrl_var_target_dist_240712_neural_bscore.pdf'), transparent=True)

# %%
acc1_list = []
acc5_list = []
for i in range(len(model_df)):
    acc_df = pd.read_csv(Path(EXP_DIR).joinpath('ctrl_var_target_dist_240712', f'run_{i:04d}', 'imagenet_acc.csv'), index_col=0)
    acc1, acc5 = acc_df['val_acc1'].max(), acc_df['val_acc5'].max()
    acc1_list.append(acc1)
    acc5_list.append(acc5)
model_df['acc1'] = acc1_list
model_df['acc5'] = acc5_list

# %%
y1 = list(model_df.groupby(['dset', 'task_set']).mean(numeric_only=True).loc[('tdw_1m_20240206', )].reindex(task_set_list)['acc1'])
y1_e = list(model_df.groupby(['dset', 'task_set']).std(numeric_only=True).loc[('tdw_1m_20240206', )].reindex(task_set_list)['acc1'])

y2 = list(model_df.groupby(['dset', 'task_set']).mean(numeric_only=True).loc[('tdw_1m_1c_n03001627_20240711', )].reindex(task_set_list)['acc1'])
y2_e = list(model_df.groupby(['dset', 'task_set']).std(numeric_only=True).loc[('tdw_1m_1c_n03001627_20240711', )].reindex(task_set_list)['acc1'])

x = [1, 2, 6, 9]
fig, ax = plt.subplots(figsize=(4.8, 3.6))

ax.errorbar(x, y1, yerr=y1_e, capsize=3, fmt='o', label='tdw_1m_20240206')
ax.errorbar(x, y2, yerr=y2_e, capsize=3, fmt='o', label='tdw_1m_1c_n03001627_20240711')

ax.legend()
ax.set_xticks(x)
ax.set_xlabel('Number of output units')
ax.set_ylabel('ImageNet-1k Top-1 Accuracy')
adjust_figure(ax)
fig.tight_layout(pad=0.5)
fig.savefig(Path(FIG_DIR).joinpath('ctrl_var_target_dist_240712_imagenet_acc1.pdf'), transparent=True)

# %%
y1 = list(model_df.groupby(['dset', 'task_set']).mean(numeric_only=True).loc[('tdw_1m_20240206', )].reindex(task_set_list)['acc5'])
y1_e = list(model_df.groupby(['dset', 'task_set']).std(numeric_only=True).loc[('tdw_1m_20240206', )].reindex(task_set_list)['acc5'])

y2 = list(model_df.groupby(['dset', 'task_set']).mean(numeric_only=True).loc[('tdw_1m_1c_n03001627_20240711', )].reindex(task_set_list)['acc5'])
y2_e = list(model_df.groupby(['dset', 'task_set']).std(numeric_only=True).loc[('tdw_1m_1c_n03001627_20240711', )].reindex(task_set_list)['acc5'])

x = [1, 2, 6, 9]
fig, ax = plt.subplots(figsize=(4.8, 3.6))

ax.errorbar(x, y1, yerr=y1_e, capsize=3, fmt='o', label='tdw_1m_20240206')
ax.errorbar(x, y2, yerr=y2_e, capsize=3, fmt='o', label='tdw_1m_1c_n03001627_20240711')

ax.legend()
ax.set_xticks(x)
ax.set_xlabel('Number of output units')
ax.set_ylabel('ImageNet-1k Top-5 Accuracy')
adjust_figure(ax)
fig.tight_layout(pad=0.5)
fig.savefig(Path(FIG_DIR).joinpath('ctrl_var_target_dist_240712_imagenet_acc5.pdf'), transparent=True)


