# %%
import pandas as pd
from pathlib import Path

from matplotlib import pyplot as plt

from config_global import EXP_DIR, FIG_DIR, DATA_DIR
import easyfigs.basicplot as bp

# %%
# compared with previous, the difference is to change the experiement name
df_mt = pd.read_csv(Path(EXP_DIR).joinpath('multi_task_tdw_1m20240206_0718', 'brainscore_results.csv'), index_col=0)
df_pt_rnd = pd.read_csv(Path(EXP_DIR).joinpath('pretrain_and_random_resnet18_0220', 'brainscore_results.csv'), index_col=0)

# %%
def replace_score(df):
    manual_score_list = []
    for i in range(len(df)):
        model_id = df.iloc[i]['model'] + '-manuallayer-rlmap1'
        benchmark_id = df.iloc[i]['benchmark_id']
        manual_score_list.append(pd.read_csv(Path(DATA_DIR).joinpath(f'{model_id}_{benchmark_id}_score.csv'), index_col=0).iloc[0]['score'])
    return df.assign(score=manual_score_list)

# %%
df_mt = replace_score(df_mt)
df_pt_rnd = replace_score(df_pt_rnd)

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
df_mt_neural = df_mt[df_mt['benchmark_region'] != 'Behavior']
df_mt_neural_agg = df_mt_neural.groupby(['exp_group', 'model'])['score'].mean().groupby('exp_group').agg(['mean', 'std'])

latent_data = list(df_mt_neural_agg['mean'].reindex(latent_task_list))
latent_error = list(df_mt_neural_agg['std'].reindex(latent_task_list))

cat_class_data = [df_mt_neural_agg['mean']['category_class'], ]
cat_class_error = [df_mt_neural_agg['std']['category_class'], ]

obj_class_data = [df_mt_neural_agg['mean']['object_class'], ]
obj_class_error = [df_mt_neural_agg['std']['object_class'], ]

mlt_data = [df_mt_neural_agg['mean']['cat_obj_class_all_latents'], ]
mlt_error = [df_mt_neural_agg['std']['cat_obj_class_all_latents'], ]

# random untrained models and imagenet pre-trained models
df_pt_rnd_neural = df_pt_rnd[df_pt_rnd['benchmark_region'] != 'Behavior']
df_pt_rnd_neural_agg = df_pt_rnd_neural.groupby(['exp_group', 'model'])['score'].mean().groupby('exp_group').agg(['mean', 'std'])

rnd_data = df_pt_rnd_neural_agg['mean']['random']
rnd_error = df_pt_rnd_neural_agg['std']['random']

# pytorch imagenet models, all the scores for the pre-trained models are the same, so the error is 0
pt_data = df_pt_rnd_neural_agg['mean']['imagenet1k_pretrain']
pt_error = df_pt_rnd_neural_agg['std']['imagenet1k_pretrain']

data_dict = {
    'Latent variable reg. (TDW-117)': {
        'x': latent_output_num_list,
        'y': latent_data,
        'error': latent_error,
    },
    'Object category cla. (TDW-117)': {
        'x': [117, ],
        'y': cat_class_data,
        'error': cat_class_error,
    },
    'Object identity cla. (TDW-117)': {
        'x': [548, ],
        'y': obj_class_data,
        'error': obj_class_error,
    },
    'All cla. + all reg. (TDW-117)': {
        'x': [674, ],
        'y': mlt_data,
        'error': mlt_error,
    },
}

fig, ax = plt.subplots(figsize=(5.5, 4.125))
bp.add_errorbars(ax, data_dict, fmt="o")
ax.scatter([1000, ], [pt_data, ], label='ImageNet-1K', color='r', marker='D')
ax.hlines(rnd_data, 1, 1000, linestyles='dashed', label='Untrained', color='k')
ax.fill_between([1, 1000], 2 * [rnd_data - rnd_error], 2 * [rnd_data + rnd_error], alpha=0.2, color='k')

ax.set_xlabel('Number of supervised output units')
ax.set_ylabel('Mean alignment score \n (V1, V2, V4, IT)')
ax.set_yticks([0.25, 0.30, 0.35, 0.40], ['0.25', '', '', '0.40'])
ax.set_xscale('log')
ax.legend(loc=(0.4, 0.2), fontsize=9)
bp.remove_top_right_spines(ax)
fig.tight_layout()
fig.savefig(Path(FIG_DIR).joinpath(f'brainscore_vs_output_num_resnset18_meanV1V2V4IT_rlmap0.pdf'), transparent=True)

# %%
for region in ['V1', 'V2', 'V4', 'IT', 'Behavior']:
    df_mt_agg = df_mt[df_mt['benchmark_region'] == region].groupby('exp_group')['score'].agg(['mean', 'std'])

    latent_data = list(df_mt_agg['mean'].reindex(latent_task_list))
    latent_error = list(df_mt_agg['std'].reindex(latent_task_list))

    cat_class_data = [df_mt_agg['mean']['category_class'], ]
    cat_class_error = [df_mt_agg['std']['category_class'], ]

    obj_class_data = [df_mt_agg['mean']['object_class'], ]
    obj_class_error = [df_mt_agg['std']['object_class'], ]

    mlt_data = [df_mt_agg['mean']['cat_obj_class_all_latents'], ]
    mlt_error = [df_mt_agg['std']['cat_obj_class_all_latents'], ]

    # random untrained models and imagenet pre-trained models
    df_pt_rnd_agg = df_pt_rnd[df_pt_rnd['benchmark_region'] == region].groupby('exp_group')['score'].agg(['mean', 'std'])
    rnd_data = df_pt_rnd_agg['mean']['random']
    rnd_error = df_pt_rnd_agg['std']['random']

    # pytorch imagenet models, all the scores for the pre-trained models are the same, so the error is 0
    pt_data = df_pt_rnd_agg['mean']['imagenet1k_pretrain']
    pt_error = df_pt_rnd_agg['std']['imagenet1k_pretrain']

    data_dict = {
        'Latent variable reg. (TDW-117)': {
            'x': latent_output_num_list,
            'y': latent_data,
            'error': latent_error,
        },
        'Object category cla. (TDW-117)': {
            'x': [117, ],
            'y': cat_class_data,
            'error': cat_class_error,
        },
        'Object identity cla. (TDW-117)': {
            'x': [548, ],
            'y': obj_class_data,
            'error': obj_class_error,
        },
        'All cla. + all reg. (TDW-117)': {
            'x': [674, ],
            'y': mlt_data,
            'error': mlt_error,
        },
    }

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    bp.add_errorbars(ax, data_dict, fmt="o")
    ax.scatter([1000, ], [pt_data, ], label='ImageNet-1K', color='r', marker='D')
    ax.hlines(rnd_data, 1, 1000, linestyles='dashed', label='Untrained', color='k')
    ax.fill_between([1, 1000], 2 * [rnd_data - rnd_error], 2 * [rnd_data + rnd_error], alpha=0.2, color='k')

    ax.set_xlabel('Number of supervised output units')
    ax.set_ylabel(f'{region} alignment score')
    # ax.set_yticks([0.25, 0.30, 0.35, 0.40], ['0.25', '', '', '0.40'])
    ax.set_xscale('log')
    bp.remove_top_right_spines(ax)
    fig.tight_layout()
    fig.savefig(Path(FIG_DIR).joinpath(f'brainscore_vs_output_num_resnset18_mean{region}_rlmap0.pdf'), transparent=True)

# %%



