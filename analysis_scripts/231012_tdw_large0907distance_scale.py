# %%
import os
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from config_global import EXP_DIR
from analysis import adjust_figure
from config_global import FIG_DIR

# %%
folder_name = '1012_analysis_tdw_large0907distance_scale'
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
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.errorbar(dataset_sizes, scores, yerr=errors, fmt='o-', label='TDW')
    ax.hlines(pret_score, dataset_sizes[0], dataset_sizes[-1], linestyles='dashed', colors='r', label='ImageNet')
    ax.hlines(rnd_score, dataset_sizes[0], dataset_sizes[-1], linestyles='dashed', colors='k', label='Random')
    ax.fill_between([dataset_sizes[0], dataset_sizes[-1]], 2 * [rnd_score - rnd_error], 2 * [rnd_score + rnd_error], alpha=0.2, color='k')
    ax.legend(loc='upper left')
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel(ylabel)
    adjust_figure(ax)
    if not os.path.exists(os.path.join(FIG_DIR, folder_name)):
        os.makedirs(os.path.join(FIG_DIR, folder_name))
    plt.savefig(os.path.join(FIG_DIR, folder_name, fig_name + '.pdf'), transparent=True)
    

# %%
# average brainscore (V1, V2, V4, IT, Behavior)
# df.groupby(['exp_group', 'model']).mean() averages over different regions
# groupby('exp_group').mean() averages over models with different seeds
all_scores = list(df.groupby(['exp_group', 'model']).mean().groupby('exp_group').mean()['score'].reindex(frac_list))
all_errors = list(df.groupby(['exp_group', 'model']).mean().groupby('exp_group').std(ddof=0)['score'].reindex(frac_list))

pret_score = df.groupby('exp_group').mean()['score']['Pre-trained']
# here use multi_task to index group name because it is the default group name
# the models are random untrained models
rnd_score = df_rnd.groupby(['exp_group', 'model']).mean().groupby('exp_group').mean()['score']['multi_task']
rnd_error = df_rnd.groupby(['exp_group', 'model']).mean().groupby('exp_group').std(ddof=0)['score']['multi_task']

bscore_scaling_plot(dataset_sizes, all_scores, all_errors, pret_score, rnd_score, rnd_error,
                    folder_name=folder_name, fig_name='scaling_brainscore_all',
                    ylabel='Mean Brain-Score \n (V1, V2, V4, IT, Behavior)')

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
# individual region scores
region_list = ['V1', 'V2', 'V4', 'IT', 'Behavior']
for region in region_list:
    df_r = df[df['benchmark_region'] == region]
    all_scores = list(df_r.groupby('exp_group').mean()['score'].reindex(frac_list))
    all_errors = list(df_r.groupby('exp_group').std(ddof=0)['score'].reindex(frac_list))

    pret_score = df_r.groupby('exp_group').mean()['score']['Pre-trained']
    # here use multi_task to index group name because it is the default group name
    # the models are random untrained models
    df_rnd_r = df_rnd[df_rnd['benchmark_region'] == region]
    rnd_score = df_rnd_r.groupby('exp_group').mean()['score']['multi_task']
    rnd_error = df_rnd_r.groupby('exp_group').std(ddof=0)['score']['multi_task']

    bscore_scaling_plot(dataset_sizes, all_scores, all_errors, pret_score, rnd_score, rnd_error,
                        folder_name=folder_name, fig_name=f'scaling_brainscore_{region}',
                        ylabel=f'{region} Score')

# %%



