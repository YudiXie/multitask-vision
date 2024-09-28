# %%
import numpy as np
import pandas as pd
from pathlib import Path

from matplotlib import pyplot as plt

from config_global import EXP_DIR, FIG_DIR, DATA_DIR
import easyfigs.basicplot as bp

# %%
df_mt = pd.read_csv(Path(EXP_DIR).joinpath('multi_task_resnet50_tdw_1m20240206_0908', 'brainscore_results.csv'), index_col=0)
df_pt_rnd = pd.read_csv(Path(EXP_DIR).joinpath('pretrain_and_random_resnet50_0220', 'brainscore_results.csv'), index_col=0)

# %%
df_nc_cat = pd.read_csv(Path(EXP_DIR).joinpath('cat_tdw_1m_nc_20240902_resnet50_0908', 'brainscore_results.csv'), index_col=0)
nc_num_list = [2, 4, 6, 8, 16]
nc_list = [f'{i}c' for i in nc_num_list]
nc_cat_groups = []
for nc in nc_list:
    nc_cat_groups.extend([nc, ] * 25)
df_nc_cat['exp_group'] = nc_cat_groups

df_imn = pd.read_csv(Path(EXP_DIR).joinpath('imagenet1k_resnet50_0908', 'brainscore_results.csv'), index_col=0)

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

# random untrained models and imagenet pre-trained models
df_pt_rnd_neural = df_pt_rnd[df_pt_rnd['benchmark_region'] != 'Behavior']
df_pt_rnd_neural_agg = df_pt_rnd_neural.groupby(['exp_group', 'model'])['score'].mean().groupby('exp_group').agg(['mean', 'std'])

rnd_data = df_pt_rnd_neural_agg['mean']['random']
rnd_error = df_pt_rnd_neural_agg['std']['random']

# pytorch imagenet models, all the scores for the pre-trained models are the same, so the error is 0
pt_data = df_pt_rnd_neural_agg['mean']['imagenet1k_pretrain']
pt_error = df_pt_rnd_neural_agg['std']['imagenet1k_pretrain']

# cateogry classification on datasets with different number of categories
df_nc_cat_neural = df_nc_cat[df_nc_cat['benchmark_region'] != 'Behavior']
df_nc_cat_neural_agg = df_nc_cat_neural.groupby(['exp_group', 'model'])['score'].mean().groupby('exp_group').agg(['mean', 'std'])
nc_data = list(df_nc_cat_neural_agg['mean'].reindex(nc_list))
nc_error = list(df_nc_cat_neural_agg['std'].reindex(nc_list))

# in hous imagenet classification models
df_imn_neural = df_imn[df_imn['benchmark_region'] != 'Behavior']
df_imn_neural_s = df_imn_neural.groupby('model')['score'].mean()
imn_data, imn_error = df_imn_neural_s.mean(), df_imn_neural_s.std()

data_dict = {
    'Spatial latents (TDW-117)': {
        'x': latent_output_num_list,
        'y': latent_data,
        'error': latent_error,
        'kwargs': {'color': '#448aff'},
    },
    'Object category (TDW-N)': {
        'x': nc_num_list,
        'y': nc_data,
        'error': nc_error,
        'kwargs': {'color': '#ffc107'},
    },
    'Object category (TDW-117)': {
        'x': [117, ],
        'y': cat_class_data,
        'error': cat_class_error,
        'kwargs': {'color': '#ff9800'},
    },
    'Object identity (TDW-117)': {
        'x': [548, ],
        'y': obj_class_data,
        'error': obj_class_error,
        'kwargs': {'color': '#DA814E'},
    },
    'All spatial + classification (TDW-117)': {
        'x': [674, ],
        'y': mlt_data,
        'error': mlt_error,
        'kwargs': {'color': '#8bc34a'},
    },
    'Object category (ImageNet-1K)': {
        'x': [1000, ],
        'y': [imn_data, ],
        'error': [imn_error, ],
        'kwargs': {'color': '#f44336'},
    },
}

fig, ax = plt.subplots(figsize=(4.8, 3.6))
bp.add_errorbars(ax, data_dict, fmt="o")
# ax.scatter([1000, ], [pt_data, ], label='ImageNet-1K', color='r', marker='D')
x_left, x_right = 0.7, 1412
ax.hlines(rnd_data, x_left, x_right, linestyles=':', label='Untrained', color='k', alpha=0.5)
ax.fill_between([x_left, x_right], 2 * [rnd_data - rnd_error], 2 * [rnd_data + rnd_error], alpha=0.1, color='k')

ax.set_xlabel('Number of supervised output units')
ax.set_ylabel('Mean neural alignment score\n(V1, V2, V4, IT)')
ax.set_yticks([0.25, 0.30, 0.35, 0.40], ['0.25', '', '', '0.40'])
ax.set_xlim(x_left, x_right)
ax.set_xscale('log')
ax.legend(loc=(0.34, 0.17), fontsize=8, title='Training targets')
bp.remove_top_right_spines(ax)
fig.tight_layout()
fig.savefig(Path(FIG_DIR).joinpath(f'brainscore_vs_output_num_resnset50_meanV1V2V4IT_batch_{batch_num}.pdf'), transparent=True, bbox_inches='tight')

# %%
batch_num = 1000000
for region in ['V1', 'V2', 'V4', 'IT', 'Behavior']:
    df_mt_agg = df_mt[(df_mt['benchmark_region'] == region) & (df_mt['batch'] == batch_num)].groupby('exp_group')['score'].agg(['mean', 'std'])

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

    # cateogry classification on datasets with different number of categories
    df_nc_cat_agg = df_nc_cat[df_nc_cat['benchmark_region'] == region].groupby('exp_group')['score'].agg(['mean', 'std'])
    nc_data = list(df_nc_cat_agg['mean'].reindex(nc_list))
    nc_error = list(df_nc_cat_agg['std'].reindex(nc_list))

    # in hous imagenet classification models
    df_imn_neural_s = df_imn[df_imn['benchmark_region'] == region]['score']
    imn_data, imn_error = df_imn_neural_s.mean(), df_imn_neural_s.std()

    data_dict = {
        'Spatial latent reg. (TDW-117)': {
            'x': latent_output_num_list,
            'y': latent_data,
            'error': latent_error,
            'kwargs': {'color': '#448aff'},
        },
        'Object category cla. (TDW-N)': {
            'x': nc_num_list,
            'y': nc_data,
            'error': nc_error,
            'kwargs': {'color': '#ffc107'},
        },
        'Object category cla. (TDW-117)': {
            'x': [117, ],
            'y': cat_class_data,
            'error': cat_class_error,
            'kwargs': {'color': '#ff9800'},
        },
        'Object identity cla. (TDW-117)': {
            'x': [548, ],
            'y': obj_class_data,
            'error': obj_class_error,
            'kwargs': {'color': '#DA814E'},
        },
        'All cla. + all reg. (TDW-117)': {
            'x': [674, ],
            'y': mlt_data,
            'error': mlt_error,
            'kwargs': {'color': '#8bc34a'},
        },
        'Object category cla. (ImageNet-1K)': {
            'x': [900, ],
            'y': [imn_data, ],
            'error': [imn_error, ],
            'kwargs': {'color': '#f44336'},
        },
    }

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    bp.add_errorbars(ax, data_dict, fmt="o")
    # ax.scatter([1000, ], [pt_data, ], label='ImageNet-1K', color='r', marker='D')
    x_left, x_right = 0.7, 1412
    ax.hlines(rnd_data, x_left, x_right, linestyles=':', label='Untrained', color='k', alpha=0.5)
    ax.fill_between([x_left, x_right], 2 * [rnd_data - rnd_error], 2 * [rnd_data + rnd_error], alpha=0.1, color='k')

    ax.set_xlabel('Number of supervised output units')
    ax.set_ylabel(f'{region} alignment score')
    # ax.set_yticks([0.25, 0.30, 0.35, 0.40], ['0.25', '', '', '0.40'])
    ax.set_xlim(x_left, x_right)
    ax.set_xscale('log')
    bp.remove_top_right_spines(ax)
    fig.tight_layout()
    fig.savefig(Path(FIG_DIR).joinpath(f'brainscore_vs_output_num_resnset50_mean{region}_batch_{batch_num}.pdf'), transparent=True, bbox_inches='tight')


# %%



