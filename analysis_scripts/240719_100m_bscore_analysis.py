# %%
import numpy as np
import pandas as pd
from pathlib import Path

from matplotlib import pyplot as plt

from config_global import EXP_DIR, FIG_DIR
from analysis import adjust_figure

# %%
frac_list = ['frac_0.0001', 'frac_0.0003', 'frac_0.001', 'frac_0.003', 'frac_0.01', 'frac_0.03', 'frac_0.1', 'frac_0.3', 'frac_1.0']
full_size = 1e8 # full size of the dataset 100M
dataset_sizes = [float(frac[5:]) * full_size for frac in frac_list]
log_dataset_sizes = np.log(dataset_sizes)

# %%
df = pd.read_csv(Path(EXP_DIR).joinpath('acal_tdw_nopret_dis_scaling_240711', 'brainscore_results.csv'), index_col=0)
neural_df = df[df['benchmark_region'] != 'Behavior']

# %%
df = pd.read_csv(Path(EXP_DIR).joinpath('acal_tdw_nopret_dis_scaling_240711', 'brainscore_results.csv'), index_col=0)
neural_df = df[df['benchmark_region'] != 'Behavior']
acal_scores = list(neural_df.groupby(['exp_group', 'model']).mean(numeric_only=True).groupby('exp_group').mean()['score'].reindex(frac_list))
acal_errors = list(neural_df.groupby(['exp_group', 'model']).mean(numeric_only=True).groupby('exp_group').std(ddof=0)['score'].reindex(frac_list))

# %%
df_pt_rnd = pd.read_csv(Path(EXP_DIR).joinpath('pretrain_and_random_resnet50_0220', 'brainscore_results.csv'), index_col=0)
df_pt_rnd_neural = df_pt_rnd[df_pt_rnd['benchmark_region'] != 'Behavior']

rnd_data = df_pt_rnd_neural.groupby(['exp_group', 'model']).mean(numeric_only=True).groupby('exp_group').mean()['score']['random']
rnd_error = df_pt_rnd_neural.groupby(['exp_group', 'model']).mean(numeric_only=True).groupby('exp_group').std(ddof=0)['score']['random']

pt_data = df_pt_rnd_neural.groupby(['exp_group', 'model']).mean(numeric_only=True).groupby('exp_group').mean()['score']['imagenet1k_pretrain']
pt_error = df_pt_rnd_neural.groupby(['exp_group', 'model']).mean(numeric_only=True).groupby('exp_group').std(ddof=0)['score']['imagenet1k_pretrain']

# %%
fig, ax = plt.subplots(figsize=(4.8, 3.6))
ax.set_xscale('log')
ax.errorbar(dataset_sizes, acal_scores, yerr=acal_errors, capsize=3, fmt='o-', label='all cat. all lat. (TDW)')
ax.scatter([1.3e6, ], [pt_data, ], label='ImageNet-1K', color='r', marker='D'),
ax.hlines(rnd_data, dataset_sizes[0], dataset_sizes[-1], linestyles='dashed', colors='k', label='Untrained')
ax.fill_between([dataset_sizes[0], dataset_sizes[-1]], 2 * [rnd_data - rnd_error], 2 * [rnd_data + rnd_error], alpha=0.2, color='k')
ax.legend(loc=(0.4, 0.2))
ax.set_xlim(5e3, 2e8)
ax.set_xlabel('Dataset size, number of images')
ax.set_ylabel('Mean Brain-Score \n (V1, V2, V4, IT)')
ax.set_yticks([0.25, 0.30, 0.35, 0.40], ['0.25', '', '', '0.40'])
adjust_figure(ax)
fig.tight_layout()
fig.savefig(Path(FIG_DIR).joinpath('0719_100m_model_scaling', '100m_model_scaling.pdf'), transparent=True)


