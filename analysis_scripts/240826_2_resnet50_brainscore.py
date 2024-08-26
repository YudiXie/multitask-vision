# %%
import numpy as np
import pandas as pd
from pathlib import Path

from matplotlib import pyplot as plt

from config_global import EXP_DIR, FIG_DIR, DATA_DIR
import easyfigs.basicplot as bp

# %%
df_mt = pd.read_csv(Path(EXP_DIR).joinpath('multi_task_resnet50_tdw_10m20240208_0802', 'brainscore_results.csv'), index_col=0)
df_pt_rnd = pd.read_csv(Path(EXP_DIR).joinpath('pretrain_and_random_resnet50_0220', 'brainscore_results.csv'), index_col=0)

# %%
# name of task groups, not individual tasks
latent_task_list = ['distance_reg', # 1
                    'translation_reg', # 2
                    'rotation_reg', # 6
                    'distance_translation', # 3
                    'distance_rotation', # 7
                    'translation_rotation', # 8
                    'distance_translation_rotation', # 9
                   ]
latent_output_num_list = [1, 2, 6, 3, 7, 8, 9]

# %%
def score_vs_output_num_plot(data_dict, 
                             pt_data, 
                             rnd_data, 
                             rnd_error,
                             ylabel, 
                             save_name,
                             yticks,
                             yticklabels,
                             legend_loc,
                             show_scaleing=False,
                             ):
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    bp.add_errorbars(ax, data_dict, fmt="o")
    ax.scatter([1000, ], [pt_data, ], label='ImageNet-1K', color='r', marker='D')
    ax.hlines(rnd_data, 1, 1000, linestyles='dashed', label='Untrained', color='k')
    ax.fill_between([1, 1000], 2 * [rnd_data - rnd_error], 2 * [rnd_data + rnd_error], alpha=0.2, color='k')
    if show_scaleing:
        ax.plot([1, 1000], [rnd_data, pt_data], 'r--')

    ax.set_xlabel('Number of CNN output units')
    ax.set_ylabel(ylabel)
    if yticks is not None and yticklabels is not None:
        ax.set_yticks(yticks, yticklabels)
    ax.set_xscale('log')
    if legend_loc:
        ax.legend(loc=legend_loc)
    bp.remove_top_right_spines(ax)
    fig.tight_layout()
    fig.savefig(Path(FIG_DIR).joinpath(f'brainscore_vs_output_num_{save_name}.pdf'), transparent=True)

# %%
df_mt_neural = df_mt[(df_mt['benchmark_region'] != 'Behavior') & (df_mt['batch'] == 1000000)]
df_mt_neural.groupby(['exp_group', 'model'])['score'].mean().groupby('exp_group').agg(['mean', 'std'])

# %%
df_pt_rnd_neural = df_pt_rnd[df_pt_rnd['benchmark_region'] != 'Behavior']
df_pt_rnd_neural.groupby(['exp_group', 'model'])['score'].mean().groupby('exp_group').agg(['mean', 'std'])

# %%
# for batch_num in df_mt['batch'].unique():
batch_num = 1000000
df_mt_neural = df_mt[(df_mt['benchmark_region'] != 'Behavior') & (df_mt['batch'] == batch_num)]
df_mt_neural_agg = df_mt_neural.groupby(['exp_group', 'model'])['score'].mean().groupby('exp_group').agg(['mean', 'std'])

latent_data = list(df_mt_neural_agg['mean'].reindex(latent_task_list))
latent_error = list(df_mt_neural_agg['std'].reindex(latent_task_list))

cat_class_data = [df_mt_neural_agg['mean']['category_class'], ]
cat_class_error = [df_mt_neural_agg['std']['category_class'], ]

obj_class_data = [df_mt_neural_agg['mean']['object_class'], ]
obj_class_error = [df_mt_neural_agg['std']['object_class'], ]

mlt_data = [df_mt_neural_agg['mean']['cat_obj_class_all_latents'], ]
mlt_error = [df_mt_neural_agg['std']['cat_obj_class_all_latents'], ]

# the models are random untrained models
df_pt_rnd_neural = df_pt_rnd[df_pt_rnd['benchmark_region'] != 'Behavior']
df_pt_rnd_neural_agg = df_pt_rnd_neural.groupby(['exp_group', 'model'])['score'].mean().groupby('exp_group').agg(['mean', 'std'])

rnd_data = df_pt_rnd_neural_agg['mean']['random']
rnd_error = df_pt_rnd_neural_agg['std']['random']

pt_data = df_pt_rnd_neural_agg['mean']['imagenet1k_pretrain']
pt_error = df_pt_rnd_neural_agg['std']['imagenet1k_pretrain']
# all the scores for the pre-trained models are the same, so the error is 0

data_dict = {
    'Spatial latent reg. (TDW)': {
        'x': latent_output_num_list,
        'y': latent_data,
        'error': latent_error,
    },
    'Object category cla. (TDW)': {
        'x': [117, ],
        'y': cat_class_data,
        'error': cat_class_error,
    },
    'Object identity cla. (TDW)': {
        'x': [548, ],
        'y': obj_class_data,
        'error': obj_class_error,
    },
    'All cla. + all reg. (TDW)': {
        'x': [674, ],
        'y': mlt_data,
        'error': mlt_error,
    },
}
score_vs_output_num_plot(data_dict, pt_data, rnd_data, rnd_error, 
                        ylabel='Mean alignment score \n (V1, V2, V4, IT)',
                        save_name=f'meanV1V2V4IT_batch_{batch_num}',
                        yticks=[0.25, 0.30, 0.35, 0.40],
                        yticklabels=['0.25', '', '', '0.40'],
                        legend_loc=(0.33, 0.18),
                        show_scaleing=True,
                        )

# %%
region_dict = {'V1': {'yticks': None, 'yticklabels': None, 'legend_loc': None},
               'V2': {'yticks': None, 'yticklabels': None, 'legend_loc': None},
               'V4': {'yticks': None, 'yticklabels': None, 'legend_loc': None},
               'IT': {'yticks': None, 'yticklabels': None, 'legend_loc': None},
               'Behavior': {'yticks': None, 'yticklabels': None, 'legend_loc': None},
               }

for region, figconfig in region_dict.items():
    df_mt_agg = df_mt[(df_mt['benchmark_region'] == region) & (df_mt['batch'] == batch_num)].groupby('exp_group')['score'].agg(['mean', 'std'])
    latent_data = list(df_mt_agg['mean'].reindex(latent_task_list))
    latent_error = list(df_mt_agg['std'].reindex(latent_task_list))

    cat_class_data = [df_mt_agg['mean']['category_class'], ]
    cat_class_error = [df_mt_agg['std']['category_class'], ]

    obj_class_data = [df_mt_agg['mean']['object_class'], ]
    obj_class_error = [df_mt_agg['std']['object_class'], ]

    mlt_data = [df_mt_agg['mean']['cat_obj_class_all_latents'], ]
    mlt_error = [df_mt_agg['std']['cat_obj_class_all_latents'], ]

    df_pt_rnd_agg = df_pt_rnd[df_pt_rnd['benchmark_region'] == region].groupby('exp_group')['score'].agg(['mean', 'std'])
    rnd_data = df_pt_rnd_agg['mean']['random']
    rnd_error = df_pt_rnd_agg['std']['random']

    pt_data = df_pt_rnd_agg['mean']['imagenet1k_pretrain']
    pt_error = df_pt_rnd_agg['std']['imagenet1k_pretrain']
    # all the scores for the pre-trained models are the same, so the error is 0

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
        'Object identity cla. (TDW)': {
            'x': [548, ],
            'y': obj_class_data,
            'error': obj_class_error,
        },
        'All cla. + all reg. (TDW)': {
            'x': [674, ],
            'y': mlt_data,
            'error': mlt_error,
        },
    }
    score_vs_output_num_plot(data_dict, pt_data, rnd_data, rnd_error,
                             ylabel=f'{region} alignment score',
                             save_name=f'{region}_batch_{batch_num}',
                             **figconfig)


# %%
# get the behavioral scores of the models when the layer is assigned to IT
df_mt_beh_final = df_mt[(df_mt['benchmark_region'] == 'Behavior') & (df_mt['batch'] == 1000000)]
behaviorit_scores = []
for model_id in df_mt_beh_final['model']:
    model_behit_id = model_id[:-14] + '-behaviorit'
    score_path = Path(DATA_DIR).joinpath(f'{model_behit_id}_Rajalingham2018public-i2n_score.csv')
    score = pd.read_csv(score_path, index_col=0)['score'][0]
    behaviorit_scores.append(score)
df_mt_beh_final_behit = df_mt_beh_final.assign(score = behaviorit_scores, benchmark_region = 'Behavior-ITlayer', error = 0, mapped_layer = 'ITlayer')

# %%
df_mt_agg = df_mt_beh_final_behit.groupby('exp_group')['score'].agg(['mean', 'std'])
latent_data = list(df_mt_agg['mean'].reindex(latent_task_list))
latent_error = list(df_mt_agg['std'].reindex(latent_task_list))

cat_class_data = [df_mt_agg['mean']['category_class'], ]
cat_class_error = [df_mt_agg['std']['category_class'], ]

obj_class_data = [df_mt_agg['mean']['object_class'], ]
obj_class_error = [df_mt_agg['std']['object_class'], ]

mlt_data = [df_mt_agg['mean']['cat_obj_class_all_latents'], ]
mlt_error = [df_mt_agg['std']['cat_obj_class_all_latents'], ]

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
    'Object identity cla. (TDW)': {
        'x': [548, ],
        'y': obj_class_data,
        'error': obj_class_error,
    },
    'All cla. + all reg. (TDW)': {
        'x': [674, ],
        'y': mlt_data,
        'error': mlt_error,
    },
}

fig, ax = plt.subplots(figsize=(4.8, 3.6))
bp.add_errorbars(ax, data_dict, fmt="o")
ax.set_xlabel('Number of CNN output units')
ax.set_ylabel('Behavioral scores from IT layer')
ax.set_xscale('log')
ax.legend(fontsize='small')
bp.remove_top_right_spines(ax)
fig.tight_layout()
fig.savefig(Path(FIG_DIR).joinpath(f'brainscore_vs_output_num_behavioral_score_from_IT.pdf'), transparent=True)

# %%
x = np.array(df_mt_beh_final['score'])
y = np.array(df_mt_beh_final_behit['score'])

fig, ax = plt.subplots(figsize=(4.8, 3.6))
ax.scatter(x, y, alpha=0.8)

ax.set_xlabel('Behavioral scores from avgpool layer')
ax.set_ylabel('Behavioral scores from IT layer')

bp.remove_top_right_spines(ax)
fig.tight_layout()
fig.savefig(Path(FIG_DIR).joinpath(f'brainscore_avgpool_vs_behavioral_score_ITlayer.pdf'), transparent=True)

# %%
df_mt_it_final = df_mt[(df_mt['benchmark_region'] == 'IT') & (df_mt['batch'] == 1000000)]
x = np.array(df_mt_it_final['score'])
y = np.array(df_mt_beh_final_behit['score'])

fig, ax = plt.subplots(figsize=(4.8, 3.6))
ax.scatter(x, y, alpha=0.8)

ax.set_xlabel('IT scores from best layer')
ax.set_ylabel('Behavioral scores from IT layer')

bp.remove_top_right_spines(ax)
fig.tight_layout()
fig.savefig(Path(FIG_DIR).joinpath(f'it_score_vs_behavioral_score_ITlayer.pdf'), transparent=True)


# check if the model ordering is correct
for i, model_id in enumerate(df_mt_it_final['model']):
    assert df_mt_beh_final_behit['model'].iloc[i] == model_id


