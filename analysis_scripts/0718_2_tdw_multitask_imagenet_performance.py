# %%
import numpy as np
import pandas as pd
from pathlib import Path

from matplotlib import pyplot as plt

from config_global import EXP_DIR, FIG_DIR
from analysis import adjust_figure
from itertools import product

# %%
task_set_list = [
    'distance_reg',
    'translation_reg',
    'rotation_reg',
    'distance_translation',
    'distance_rotation',
    'translation_rotation',
    'distance_translation_rotation',
    'category_class',
    'object_class',
    'cat_obj_class_all_latents',
]
seed_list = [0, 1, 2]
model_dict = {'model': [], 'task_set': [], 'seed': []}
for i_s, state in enumerate(product(task_set_list, seed_list)):
    task_set, seed = state
    model_dict['model'].append(f'multi_task_tdw_1m20240206_nopret_0214-resnet18-{i_s}')
    model_dict['task_set'].append(task_set)
    model_dict['seed'].append(seed)
model_df = pd.DataFrame(model_dict)

# %%
model_df

# %%
acc1_list = []
acc5_list = []
for i in range(len(model_df)):
    acc_df = pd.read_csv(Path(EXP_DIR).joinpath('multi_task_tdw_1m20240206_nopret_0214', f'run_{i:04d}', 'imagenet_acc.csv'), index_col=0)
    acc1, acc5 = acc_df['val_acc1'].max(), acc_df['val_acc5'].max()
    acc1_list.append(acc1)
    acc5_list.append(acc5)
model_df['acc1'] = acc1_list
model_df['acc5'] = acc5_list

# %%
y1 = list(model_df.groupby('task_set').mean(numeric_only=True).reindex(task_set_list)['acc1'])
y1_e = list(model_df.groupby('task_set').std(numeric_only=True).reindex(task_set_list)['acc1'])

y2 = list(model_df.groupby('task_set').mean(numeric_only=True).reindex(task_set_list)['acc5'])
y2_e = list(model_df.groupby('task_set').std(numeric_only=True).reindex(task_set_list)['acc5'])

# %%
output_num_list = [1, 2, 6, 3, 7, 8, 9, 117, 548, 674]
fig, ax = plt.subplots(figsize=(4.8, 3.6))
ax.errorbar(output_num_list, y1, yerr=y1_e, fmt='o', capsize=3)
ax.set_xscale('log')
ax.set_xlabel('Number of output units')
ax.set_ylabel('ImageNet Top-1 accuracy')
ax.set_ylim(0, None)
adjust_figure(ax)
fig.tight_layout()
fig.savefig(Path(FIG_DIR).joinpath('multi_task_tdw_1m20240206_nopret_0214_imagenet_acc_vs_output_num.pdf'), transparent=True)

# %%



