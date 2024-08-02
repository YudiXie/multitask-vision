import os
import pandas as pd

from matplotlib import pyplot as plt

from config_global import EXP_DIR
from analysis import scatter_errorbar

# compared with previous, the difference is to change the experiement name, and there is no cat experiment
df_mt = pd.read_csv(os.path.join(EXP_DIR, 'multi_task_tdw_large20230907_0919', 'brainscore_results.csv'), index_col=0)
df_rnd = pd.read_csv(os.path.join(EXP_DIR, 'random_models0630', 'brainscore_results.csv'), index_col=0)

# name of task groups, not individual tasks
latent_task_list = ['distance_reg', # 1
                    'translation_reg', # 2
                    'rotation_reg', # 3
                    'distance_translation', # 3
                    'distance_rotation', # 4
                    'translation_rotation', # 5
                    'distance_translation_rotation', # 6
                   # 'categorization', # 8
                   # 'multi_task_wo_object_class', # 14
                   # 'multi_task' # 78
                   ]
latent_output_num_list = [1, 2, 3, 3, 4, 5, 6]

# compared with 0914, the difference is
# (1) change image save folder to 0924_analysis_tdw_large0907
# (2) catergorization experiment is read from the multi-tasks training group
# (3) catergorization has 117 output units, multi_task has 117 + 6 output units not 14, so moved to 0
latent_data = list(df_mt.groupby(['exp_group', 'model']).mean().groupby('exp_group').mean()['score'].reindex(latent_task_list))
latent_error = list(df_mt.groupby(['exp_group', 'model']).mean().groupby('exp_group').std(ddof=0)['score'].reindex(latent_task_list))

cat_data = [df_mt.groupby(['exp_group', 'model']).mean().groupby('exp_group').mean()['score']['categorization'], ]
cat_error = [df_mt.groupby(['exp_group', 'model']).mean().groupby('exp_group').std(ddof=0)['score']['categorization'], ]

mlt_data = [df_mt.groupby(['exp_group', 'model']).mean().groupby('exp_group').mean()['score']['multi_task'], ]
mlt_error = [df_mt.groupby(['exp_group', 'model']).mean().groupby('exp_group').std(ddof=0)['score']['multi_task'], ]

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
    'cat_tasks': {
        'x': [0, ],
        'y': cat_data,
        'error': cat_error,
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
}

add_plots = [
    lambda: plt.hlines(pret_score, 0, 15, linestyles='dashed', label='Pre-trained'),
    ]
scatter_errorbar(data_dict,
                 x_label='Number of output units',
                 y_label='Mean brain score \n (V1, V2, V4, IT, Behavior)',
                 additional_plots=add_plots,
                 folder_name='0924_analysis_tdw_large0907',
                 fig_name='brainscore_vs_output_num_all',
                 )

# compared with 0914, the difference is 
# (1) change image save folder to 0924_analysis_tdw_large0907
# (2) catergorization experiment is read from the multi-tasks training group
# (3) catergorization has 117 output units, multi_task has 117 + 6 output units not 14, so moved to 0
latent_data = list(df_mt[df_mt['benchmark_region'] != 'Behavior'].groupby(['exp_group', 'model']).mean().groupby('exp_group').mean()['score'].reindex(latent_task_list))
latent_error = list(df_mt[df_mt['benchmark_region'] != 'Behavior'].groupby(['exp_group', 'model']).mean().groupby('exp_group').std(ddof=0)['score'].reindex(latent_task_list))

cat_data = [df_mt[df_mt['benchmark_region'] != 'Behavior'].groupby(['exp_group', 'model']).mean().groupby('exp_group').mean()['score']['categorization'], ]
cat_error = [df_mt[df_mt['benchmark_region'] != 'Behavior'].groupby(['exp_group', 'model']).mean().groupby('exp_group').std(ddof=0)['score']['categorization'], ]

mlt_data = [df_mt[df_mt['benchmark_region'] != 'Behavior'].groupby(['exp_group', 'model']).mean().groupby('exp_group').mean()['score']['multi_task'], ]
mlt_error = [df_mt[df_mt['benchmark_region'] != 'Behavior'].groupby(['exp_group', 'model']).mean().groupby('exp_group').std(ddof=0)['score']['multi_task'], ]

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
    'cat_tasks': {
        'x': [0, ],
        'y': cat_data,
        'error': cat_error,
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
}

add_plots = [
    lambda: plt.hlines(pret_score, 0, 15, linestyles='dashed', label='Pre-trained'),
    ]
scatter_errorbar(data_dict,
                 x_label='Number of output units',
                 y_label='Mean brain score \n (V1, V2, V4, IT)',
                 additional_plots=add_plots,
                 folder_name='0924_analysis_tdw_large0907',
                 fig_name='brainscore_vs_output_num_wo_behavior',
                 )

# compared with 0914, the difference is
# (1) change image save folder to 0924_analysis_tdw_large0907
# (2) catergorization experiment is read from the multi-tasks training group
# (3) catergorization has 117 output units, multi_task has 117 + 6 output units not 14, so moved to 0
region_list = ['V1', 'V2', 'V4', 'IT', 'Behavior']
for region in region_list:
    df_mt_r = df_mt[df_mt['benchmark_region'] == region]
    latent_data = list(df_mt_r.groupby('exp_group').mean()['score'].reindex(latent_task_list))
    latent_error = list(df_mt_r.groupby('exp_group').std(ddof=0)['score'].reindex(latent_task_list))

    df_cat_r = df_mt[df_mt['benchmark_region'] == region]
    cat_data = [df_cat_r.groupby('exp_group').mean()['score']['categorization'], ]
    cat_error = [df_cat_r.groupby('exp_group').std(ddof=0)['score']['categorization'], ]

    mlt_data = [df_mt_r.groupby('exp_group').mean()['score']['multi_task'], ]
    mlt_error = [df_mt_r.groupby('exp_group').std(ddof=0)['score']['multi_task'], ]

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
        'cat_tasks': {
            'x': [0, ],
            'y': cat_data,
            'error': cat_error,
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
    }

    add_plots = [
        lambda: plt.hlines(pret_score, 0, 15, linestyles='dashed', label='Pre-trained'),
        ]
    scatter_errorbar(data_dict, 
                     x_label='Number of output units', 
                     y_label=f'{region} score', 
                     additional_plots=add_plots,
                     folder_name='0924_analysis_tdw_large0907',
                     fig_name=f'brainscore_vs_output_num_{region}',
                    )
