# %%
import numpy as np
import pandas as pd
from pathlib import Path

from matplotlib import pyplot as plt

from config_global import EXP_DIR, FIG_DIR
import easyfigs.basicplot as bp

# %%
frac_list = ['frac_0.0001', 'frac_0.0003', 'frac_0.001', 'frac_0.003', 'frac_0.01', 'frac_0.03', 'frac_0.1', 'frac_0.3', 'frac_1.0']
full_size = 1e8 # full size of the dataset 100M
dataset_sizes = [float(frac[5:]) * full_size for frac in frac_list]
dataset_sizes[-1] = 96e6 # last dataset size is 96M becuase of maxium training batch limit
log_dataset_sizes = np.log(dataset_sizes)

# %%
df = pd.read_csv(Path(EXP_DIR).joinpath('resnet50_tdw100m_scaling_240822', 'brainscore_results.csv'), index_col=0)

task_list = []
for task in ['distance_reg', 'translation_reg', 'rotation_reg', 'category_class', 'cat_obj_class_all_latents']:
    task_list.extend([task, ] * 3 * 5)
all_task_list = task_list * len(frac_list)

all_reversed_frac_list = []
for frac in reversed(frac_list):
    all_reversed_frac_list.extend([frac] * 3 * 5 * 5)

df['train_task'] = all_task_list
df['frac'] = all_reversed_frac_list

# %%
neural_df = df[df['benchmark_region'] != 'Behavior']
plot_data = {}
for task in task_list:
    task_neural_df = neural_df[neural_df['train_task'] == task]
    task_agg = task_neural_df.groupby(['frac', 'model'])['score'].mean().groupby('frac').agg(['mean', 'std']).reindex(frac_list)
    plot_data[task] = task_agg['mean'].to_numpy(), task_agg['std'].to_numpy()

# %%
df_pt_rnd = pd.read_csv(Path(EXP_DIR).joinpath('pretrain_and_random_resnet50_0220', 'brainscore_results.csv'), index_col=0)
# the models are random untrained models
df_pt_rnd_neural = df_pt_rnd[df_pt_rnd['benchmark_region'] != 'Behavior']
df_pt_rnd_neural_agg = df_pt_rnd_neural.groupby(['exp_group', 'model'])['score'].mean().groupby('exp_group').agg(['mean', 'std'])

rnd_data = df_pt_rnd_neural_agg['mean']['random']
rnd_error = df_pt_rnd_neural_agg['std']['random']

pt_data = df_pt_rnd_neural_agg['mean']['imagenet1k_pretrain']
pt_error = df_pt_rnd_neural_agg['std']['imagenet1k_pretrain']
# all the scores for the pre-trained models are the same, so the error is 0

# %%
legend_names = {
    'distance_reg': 'Distance',
    'translation_reg': 'Translation',
    'rotation_reg': 'Rotation',
    'category_class': 'Obj. category',
    'cat_obj_class_all_latents': 'All spatial + classification',
}

# %%
color_list = ['448aff', '1565c0', '009688', 'ffc107', 'ff9800', 'f44336', '707078']
fig, ax = plt.subplots(figsize=(4.8, 3.6))
ax.set_xscale('log')
for key, value in plot_data.items():
    ax.errorbar(dataset_sizes, value[0], yerr=value[1], capsize=3, fmt='o-', label=legend_names[key], color='#' + color_list.pop(0))
ax.scatter([1.3e6, ], [pt_data, ], label='ImageNet-1K', color='#' + color_list.pop(0), marker='D'),
ax.hlines(rnd_data, dataset_sizes[0], dataset_sizes[-1], linestyles='dashed', colors='k', label='Untrained')
ax.fill_between([dataset_sizes[0], dataset_sizes[-1]], 2 * [rnd_data - rnd_error], 2 * [rnd_data + rnd_error], alpha=0.2, color='k')
ax.legend(loc=(0.43, 0.17), fontsize='small', title='Training task')
ax.set_xlim(5e3, 2e8)
ax.set_xlabel('Dataset size, number of images')
ax.set_ylabel('Mean Brain-Score \n (V1, V2, V4, IT)')
ax.set_yticks([0.25, 0.30, 0.35, 0.40], ['0.25', '', '', '0.40'])
bp.remove_top_right_spines(ax)
fig.tight_layout()
fig.savefig(Path(FIG_DIR).joinpath('100m_model_scaling.pdf'), transparent=True)

# %%



