# %%
import os
import numpy as np
import pandas as pd

import scipy.stats
from matplotlib import pyplot as plt

from config_global import EXP_DIR, FIG_DIR
from analysis import scatter_errorbar, adjust_figure

# %%
# compared with previous, the difference is to change the experiement name, and there is no cat experiment
df_mt = pd.read_csv(os.path.join(EXP_DIR, 'multi_task_tdw_large20230907_nopret_0925', 'brainscore_results.csv'), index_col=0)
df_rnd = pd.read_csv(os.path.join(EXP_DIR, 'random_models0630', 'brainscore_results.csv'), index_col=0)

# %%
# name of task groups, not individual tasks
latent_task_list = ['distance_reg', # 1
                    'translation_reg', # 2
                    'rotation_reg', # 3
                    'distance_translation', # 3
                    'distance_rotation', # 4
                    'translation_rotation', # 5
                    'distance_translation_rotation', # 6
                   ]
latent_output_num_list = [1, 2, 3, 3, 4, 5, 6]

# %%
df_mt['exp_group'].unique()

# %%
pd.read_csv(os.path.join(EXP_DIR, 'multi_task_tdw_large20230907_0919', 'brainscore_results.csv'), index_col=0)['exp_group'].unique()

# %%
# compared with 0924, the difference is 
# (1) change image save folder to 1012_analysis_tdw_large0907_nopret
# (2) addition of category_class, object_class, and cat_obj_class_all_latents
# (3) 'multi_task' is now 'cat_obj_class_all_latents'
df_mt_neural = df_mt[df_mt['benchmark_region'] != 'Behavior']
latent_data = list(df_mt_neural.groupby(['exp_group', 'model']).mean().groupby('exp_group').mean()['score'].reindex(latent_task_list))
latent_error = list(df_mt_neural.groupby(['exp_group', 'model']).mean().groupby('exp_group').std(ddof=0)['score'].reindex(latent_task_list))

cat_class_data = [df_mt_neural.groupby(['exp_group', 'model']).mean().groupby('exp_group').mean()['score']['category_class'], ]
cat_class_error = [df_mt_neural.groupby(['exp_group', 'model']).mean().groupby('exp_group').std(ddof=0)['score']['category_class'], ]

obj_class_data = [df_mt_neural.groupby(['exp_group', 'model']).mean().groupby('exp_group').mean()['score']['object_class'], ]
obj_class_error = [df_mt_neural.groupby(['exp_group', 'model']).mean().groupby('exp_group').std(ddof=0)['score']['object_class'], ]

mlt_data = [df_mt_neural.groupby(['exp_group', 'model']).mean().groupby('exp_group').mean()['score']['cat_obj_class_all_latents'], ]
mlt_error = [df_mt_neural.groupby(['exp_group', 'model']).mean().groupby('exp_group').std(ddof=0)['score']['cat_obj_class_all_latents'], ]

# here use multi_task to index group name because it is the default group name
# the models are random untrained models
df_rnd_neural = df_rnd[df_rnd['benchmark_region'] != 'Behavior']
rnd_data = [df_rnd_neural.groupby(['exp_group', 'model']).mean().groupby('exp_group').mean()['score']['multi_task'], ]
rnd_error = [df_rnd_neural.groupby(['exp_group', 'model']).mean().groupby('exp_group').std(ddof=0)['score']['multi_task'], ]

pret_score = df_mt_neural.groupby('exp_group').mean()['score']['Pre-trained']

data_dict = {
    'Latent variable reg. (TDW)': {
        'x': latent_output_num_list,
        'y': latent_data,
        'error': latent_error,
    },
    'Object category cla. (TDW)': {
        'x': [117, ],
        'y': cat_class_data,
        'error': cat_class_error,
    },
    'Object model cla. (TDW)': {
        'x': [587, ],
        'y': obj_class_data,
        'error': obj_class_error,
    },
    'All cla. + all reg. (TDW)': {
        'x': [710, ],
        'y': mlt_data,
        'error': mlt_error,
    },
}

add_plots = [
    lambda: plt.scatter([1000, ], [pret_score, ], label='ImageNet-1K', color='r', marker='D'),
    lambda: plt.hlines(rnd_data[0], 1, 1000, linestyles='dashed', label='Untrained', color='k'),
    lambda: plt.fill_between([1, 1000], 2 * [rnd_data[0] - rnd_error[0]], 2 * [rnd_data[0] + rnd_error[0]], alpha=0.2, color='k'),
    ]
scatter_errorbar(data_dict,
                 x_label='Number of CNN output units',
                 y_label='Mean Brain-Score \n (V1, V2, V4, IT)',
                 additional_plots=add_plots,
                 folder_name='1110_cosyne_submission',
                 fig_name='brainscore_vs_output_num_wo_behavior',
                 log_scale=True,
                 )


# %%
group1_name = 'distance_rotation'
df_dr = df_mt_neural[df_mt_neural['exp_group'] == group1_name]
l1 = list(df_dr.groupby('model').mean()['score'])

# %%
group2_name = 'category_class'
df_comp = df_mt_neural[df_mt_neural['exp_group'] == group2_name]
l2 = list(df_comp.groupby('model').mean()['score'])
p_value = scipy.stats.mannwhitneyu(l1, l2).pvalue
print(f'{group1_name}: {np.mean(l1):.3} vs. {group2_name}: {np.mean(l2):.3}')
print(f'Mann-Whitney U test p-value: {p_value}')

# %%
group2_name = 'object_class'
df_comp = df_mt_neural[df_mt_neural['exp_group'] == group2_name]
l2 = list(df_comp.groupby('model').mean()['score'])
p_value = scipy.stats.mannwhitneyu(l1, l2).pvalue
print(f'{group1_name}: {np.mean(l1):.3} vs. {group2_name}: {np.mean(l2):.3}')
print(f'Mann-Whitney U test p-value: {p_value}')

# %%
group2_name = 'cat_obj_class_all_latents'
df_comp = df_mt_neural[df_mt_neural['exp_group'] == group2_name]
l2 = list(df_comp.groupby('model').mean()['score'])
p_value = scipy.stats.mannwhitneyu(l1, l2).pvalue
print(f'{group1_name}: {np.mean(l1):.3} vs. {group2_name}: {np.mean(l2):.3}')
print(f'Mann-Whitney U test p-value: {p_value}')

# %%
print(f'Pretrain score: {pret_score}')

# %%
print(f'best latent model / ImageNet-1K: {np.mean(l2) / pret_score}')

# %%
group1_name = 'distance_reg'
df_dr = df_mt_neural[df_mt_neural['exp_group'] == group1_name]
l1 = list(df_dr.groupby('model').mean()['score'])
print(f'{group1_name}: {np.mean(l1):.3}')

# %%
print(f'distance_reg / all class all latents: {0.368 / 0.388}')

# %% [markdown]
# ### Scaling analysis

# %%
folder_name = '1110_cosyne_submission'
df = pd.read_csv(os.path.join(EXP_DIR, 'multi_task_tdw_large20230907_nopret_dis_scaling_0925', 'brainscore_results.csv'), index_col=0)
df_rnd = pd.read_csv(os.path.join(EXP_DIR, 'random_models0630', 'brainscore_results.csv'), index_col=0)

# %%
df['exp_group'].unique()

# %%
frac_list = ['frac_0.001', 'frac_0.003', 'frac_0.01', 'frac_0.03', 'frac_0.1', 'frac_0.3', 'frac_1.0']
full_size = 1.35e6 # full size of the dataset
dataset_sizes = [float(frac[5:]) * full_size for frac in frac_list]
log_dataset_sizes = np.log(dataset_sizes)

# %%
def bscore_scaling_plot(dataset_sizes, 
                        scores, 
                        errors, 
                        pret_score, 
                        rnd_score, 
                        rnd_error,
                        folder_name,
                        fig_name,
                        ylabel,
                        ):
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.set_xscale('log')
    ax.errorbar(dataset_sizes, scores, yerr=errors, capsize=3, fmt='o-', label='Distance regression (TDW)')
    ax.scatter([dataset_sizes[-1], ], [pret_score, ], label='ImageNet-1K', color='r', marker='D'),
    ax.hlines(rnd_score, dataset_sizes[0], dataset_sizes[-1], linestyles='dashed', colors='k', label='Untrained')
    ax.fill_between([dataset_sizes[0], dataset_sizes[-1]], 2 * [rnd_score - rnd_error], 2 * [rnd_score + rnd_error], alpha=0.2, color='k')
    ax.legend(loc='upper left')
    ax.set_xlabel('Dataset size, number of images')
    ax.set_ylabel(ylabel)
    adjust_figure(ax)
    if not os.path.exists(os.path.join(FIG_DIR, folder_name)):
        os.makedirs(os.path.join(FIG_DIR, folder_name))
    plt.savefig(os.path.join(FIG_DIR, folder_name, fig_name + '.pdf'), transparent=True)

# %%
# average brainscore without behavior (V1, V2, V4, IT)
neural_df = df[df['benchmark_region'] != 'Behavior']
all_scores = list(neural_df.groupby(['exp_group', 'model']).mean().groupby('exp_group').mean()['score'].reindex(frac_list))
all_errors = list(neural_df.groupby(['exp_group', 'model']).mean().groupby('exp_group').std(ddof=0)['score'].reindex(frac_list))

pret_score = neural_df.groupby('exp_group').mean()['score']['Pre-trained']
# here use multi_task to index group name because it is the default group name
# the models are random untrained models
rnd_score = df_rnd[df_rnd['benchmark_region'] != 'Behavior'].groupby(['exp_group', 'model']).mean().groupby('exp_group').mean()['score']['multi_task']
rnd_error = df_rnd[df_rnd['benchmark_region'] != 'Behavior'].groupby(['exp_group', 'model']).mean().groupby('exp_group').std(ddof=0)['score']['multi_task']

bscore_scaling_plot(dataset_sizes, all_scores, all_errors, pret_score, rnd_score, rnd_error,
                    folder_name=folder_name, fig_name='scaling_brainscore_neural',
                    ylabel='Mean Brain-Score \n (V1, V2, V4, IT)')

# %%



