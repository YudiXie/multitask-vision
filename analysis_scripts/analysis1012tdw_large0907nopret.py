# %%
%load_ext autoreload
%autoreload 2

%cd ..
%pwd

# %%
import os
import pandas as pd

from matplotlib import pyplot as plt

from config_global import EXP_DIR
from analysis import scatter_errorbar

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
latent_data = list(df_mt.groupby(['exp_group', 'model']).mean().groupby('exp_group').mean()['score'].reindex(latent_task_list))
latent_error = list(df_mt.groupby(['exp_group', 'model']).mean().groupby('exp_group').std(ddof=0)['score'].reindex(latent_task_list))

cat_class_data = [df_mt.groupby(['exp_group', 'model']).mean().groupby('exp_group').mean()['score']['category_class'], ]
cat_class_error = [df_mt.groupby(['exp_group', 'model']).mean().groupby('exp_group').std(ddof=0)['score']['category_class'], ]

obj_class_data = [df_mt.groupby(['exp_group', 'model']).mean().groupby('exp_group').mean()['score']['object_class'], ]
obj_class_error = [df_mt.groupby(['exp_group', 'model']).mean().groupby('exp_group').std(ddof=0)['score']['object_class'], ]

mlt_data = [df_mt.groupby(['exp_group', 'model']).mean().groupby('exp_group').mean()['score']['cat_obj_class_all_latents'], ]
mlt_error = [df_mt.groupby(['exp_group', 'model']).mean().groupby('exp_group').std(ddof=0)['score']['cat_obj_class_all_latents'], ]

# here use multi_task to index group name because it is the default group name
# the models are random untrained models
rnd_data = [df_rnd.groupby(['exp_group', 'model']).mean().groupby('exp_group').mean()['score']['multi_task'], ]
rnd_error = [df_rnd.groupby(['exp_group', 'model']).mean().groupby('exp_group').std(ddof=0)['score']['multi_task'], ]

pret_score = df_mt.groupby('exp_group').mean()['score']['Pre-trained']

data_dict = {
    'latent_tasks': {
        'x': latent_output_num_list,
        'y': latent_data,
        'error': latent_error,
    },
    'cat_class': {
        'x': [0, ],
        'y': cat_class_data,
        'error': cat_class_error,
    },
    'multi_task': {
        'x': [0, ],
        'y': mlt_data,
        'error': mlt_error,
    },
    'random': {
        'x': [0, ],
        'y': rnd_data,
        'error': rnd_error,
    },
    'obj_class': {
        'x': [0, ],
        'y': obj_class_data,
        'error': obj_class_error,
    },
}

add_plots = [
    lambda: plt.hlines(pret_score, 0, 15, linestyles='dashed', label='Pre-trained'),
    ]
scatter_errorbar(data_dict,
                 x_label='Number of output units',
                 y_label='Mean brain score \n (V1, V2, V4, IT, Behavior)',
                 additional_plots=add_plots,
                 folder_name='1012_analysis_tdw_large0907_nopret',
                 fig_name='brainscore_vs_output_num_all',
                 )

# %%
# compared with 0924, the difference is 
# (1) change image save folder to 1012_analysis_tdw_large0907_nopret
# (2) addition of category_class, object_class, and cat_obj_class_all_latents
# (3) 'multi_task' is now 'cat_obj_class_all_latents'
latent_data = list(df_mt[df_mt['benchmark_region'] != 'Behavior'].groupby(['exp_group', 'model']).mean().groupby('exp_group').mean()['score'].reindex(latent_task_list))
latent_error = list(df_mt[df_mt['benchmark_region'] != 'Behavior'].groupby(['exp_group', 'model']).mean().groupby('exp_group').std(ddof=0)['score'].reindex(latent_task_list))

cat_class_data = [df_mt[df_mt['benchmark_region'] != 'Behavior'].groupby(['exp_group', 'model']).mean().groupby('exp_group').mean()['score']['category_class'], ]
cat_class_error = [df_mt[df_mt['benchmark_region'] != 'Behavior'].groupby(['exp_group', 'model']).mean().groupby('exp_group').std(ddof=0)['score']['category_class'], ]

obj_class_data = [df_mt[df_mt['benchmark_region'] != 'Behavior'].groupby(['exp_group', 'model']).mean().groupby('exp_group').mean()['score']['object_class'], ]
obj_class_error = [df_mt[df_mt['benchmark_region'] != 'Behavior'].groupby(['exp_group', 'model']).mean().groupby('exp_group').std(ddof=0)['score']['object_class'], ]

mlt_data = [df_mt[df_mt['benchmark_region'] != 'Behavior'].groupby(['exp_group', 'model']).mean().groupby('exp_group').mean()['score']['cat_obj_class_all_latents'], ]
mlt_error = [df_mt[df_mt['benchmark_region'] != 'Behavior'].groupby(['exp_group', 'model']).mean().groupby('exp_group').std(ddof=0)['score']['cat_obj_class_all_latents'], ]

# here use multi_task to index group name because it is the default group name
# the models are random untrained models
rnd_data = [df_rnd[df_rnd['benchmark_region'] != 'Behavior'].groupby(['exp_group', 'model']).mean().groupby('exp_group').mean()['score']['multi_task'], ]
rnd_error = [df_rnd[df_rnd['benchmark_region'] != 'Behavior'].groupby(['exp_group', 'model']).mean().groupby('exp_group').std(ddof=0)['score']['multi_task'], ]

pret_score = df_mt[df_mt['benchmark_region'] != 'Behavior'].groupby('exp_group').mean()['score']['Pre-trained']

data_dict = {
    'latent_tasks': {
        'x': latent_output_num_list,
        'y': latent_data,
        'error': latent_error,
    },
    'cat_class': {
        'x': [0, ],
        'y': cat_class_data,
        'error': cat_class_error,
    },
    'multi_task': {
        'x': [0, ],
        'y': mlt_data,
        'error': mlt_error,
    },
    'random': {
        'x': [0, ],
        'y': rnd_data,
        'error': rnd_error,
    },
    'obj_class': {
        'x': [0, ],
        'y': obj_class_data,
        'error': obj_class_error,
    },
}

add_plots = [
    lambda: plt.hlines(pret_score, 0, 15, linestyles='dashed', label='Pre-trained'),
    ]
scatter_errorbar(data_dict,
                 x_label='Number of output units',
                 y_label='Mean brain score \n (V1, V2, V4, IT)',
                 additional_plots=add_plots,
                 folder_name='1012_analysis_tdw_large0907_nopret',
                 fig_name='brainscore_vs_output_num_wo_behavior',
                 )


# %%
# compared with 0924, the difference is
# (1) change image save folder to 1012_analysis_tdw_large0907_nopret
# (2) addition of category_class, object_class, and cat_obj_class_all_latents
# (3) 'multi_task' is now 'cat_obj_class_all_latents'

region_list = ['V1', 'V2', 'V4', 'IT', 'Behavior']
for region in region_list:
    df_mt_r = df_mt[df_mt['benchmark_region'] == region]
    latent_data = list(df_mt_r.groupby('exp_group').mean()['score'].reindex(latent_task_list))
    latent_error = list(df_mt_r.groupby('exp_group').std(ddof=0)['score'].reindex(latent_task_list))

    cat_class_data = [df_mt_r.groupby('exp_group').mean()['score']['category_class'], ]
    cat_class_error = [df_mt_r.groupby('exp_group').std(ddof=0)['score']['category_class'], ]

    obj_class_data = [df_mt_r.groupby('exp_group').mean()['score']['object_class'], ]
    obj_class_error = [df_mt_r.groupby('exp_group').std(ddof=0)['score']['object_class'], ]

    mlt_data = [df_mt_r.groupby('exp_group').mean()['score']['cat_obj_class_all_latents'], ]
    mlt_error = [df_mt_r.groupby('exp_group').std(ddof=0)['score']['cat_obj_class_all_latents'], ]

    # here use multi_task to index group name because it is the default group name
    # the models are random untrained models
    df_rnd_r = df_rnd[df_rnd['benchmark_region'] == region]
    rnd_data = [df_rnd_r.groupby('exp_group').mean()['score']['multi_task'], ]
    rnd_error = [df_rnd_r.groupby('exp_group').std(ddof=0)['score']['multi_task'], ]

    pret_score = df_mt_r.groupby('exp_group').mean()['score']['Pre-trained']

    data_dict = {
        'latent_tasks': {
            'x': latent_output_num_list,
            'y': latent_data,
            'error': latent_error,
        },
        'cat_class': {
            'x': [0, ],
            'y': cat_class_data,
            'error': cat_class_error,
        },
        'multi_task': {
            'x': [0, ],
            'y': mlt_data,
            'error': mlt_error,
        },
        'random': {
            'x': [0, ],
            'y': rnd_data,
            'error': rnd_error,
        },
        'obj_class': {
            'x': [0, ],
            'y': obj_class_data,
            'error': obj_class_error,
        },
    }
    
    add_plots = [
        lambda: plt.hlines(pret_score, 0, 15, linestyles='dashed', label='Pre-trained'),
        ]
    scatter_errorbar(data_dict, 
                     x_label='Number of output units', 
                     y_label=f'{region} score', 
                     additional_plots=add_plots,
                     folder_name='1012_analysis_tdw_large0907_nopret',
                     fig_name=f'brainscore_vs_output_num_{region}',
                    )


# %%



