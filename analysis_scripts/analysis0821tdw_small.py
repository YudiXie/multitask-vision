import os
import pandas as pd

from matplotlib import pyplot as plt

from config_global import EXP_DIR
from analysis import scatter_errorbar


df_mt = pd.read_csv(os.path.join(EXP_DIR, 'multi_task_tdw_0817', 'brainscore_results.csv'), index_col=0)
df_cat = pd.read_csv(os.path.join(EXP_DIR, 'cat_diff_tdw_0820', 'brainscore_results.csv'), index_col=0)
df_rnd = pd.read_csv(os.path.join(EXP_DIR, 'random_models0630', 'brainscore_results.csv'), index_col=0)

# name of task groups, not individual tasks
latent_task_list = ['depth_reg', # 1
                    'translation_reg', # 2
                    'rotation_reg', # 3
                    'depth_translation', # 3
                    'depth_rotation', # 4
                    'translation_rotation', # 5
                    'depth_translation_rotation', # 6
                   # 'categorization', # 8
                   # 'multi_task_wo_object_class', # 14
                   # 'multi_task' # 78
                   ]
latent_output_num_list = [1, 2, 3, 3, 4, 5, 6]

cat_task_list = ['cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8']
cat_output_num_list = [2, 3, 4, 5, 6, 7, 8]

# compared with 0705, the difference is (1) change image save folder to 0821_analysis_tdw_small, (2) mlt_data: multi_task_wo_object_class is now multi_task
latent_data = list(df_mt.groupby(['exp_group', 'model']).mean().groupby('exp_group').mean()['score'].reindex(latent_task_list))
latent_error = list(df_mt.groupby(['exp_group', 'model']).mean().groupby('exp_group').std(ddof=0)['score'].reindex(latent_task_list))

cat_data = list(df_cat.groupby(['exp_group', 'model']).mean().groupby('exp_group').mean()['score'].reindex(cat_task_list))
cat_error = list(df_cat.groupby(['exp_group', 'model']).mean().groupby('exp_group').std(ddof=0)['score'].reindex(cat_task_list))

mlt_data = df_mt.groupby(['exp_group', 'model']).mean().groupby('exp_group').mean()['score']['multi_task']
mlt_data = [mlt_data, ]
mlt_error = df_mt.groupby(['exp_group', 'model']).mean().groupby('exp_group').std(ddof=0)['score']['multi_task']
mlt_error = [mlt_error, ]

# here use multi_task to index group name because it is the default group name
# the models are random untrained models
rnd_data = df_rnd.groupby(['exp_group', 'model']).mean().groupby('exp_group').mean()['score']['multi_task']
rnd_data = [rnd_data, ]
rnd_error = df_rnd.groupby(['exp_group', 'model']).mean().groupby('exp_group').std(ddof=0)['score']['multi_task']
rnd_error = [rnd_error, ]

pret_score = df_mt.groupby('exp_group').mean()['score']['Pre-trained']

data_dict = {
    'latent_tasks': {
        'x': latent_output_num_list,
        'y': latent_data,
        'error': latent_error
    },
    'cat_tasks': {
        'x': cat_output_num_list,
        'y': cat_data,
        'error': cat_error
    },
    'multi_task': {
        'x': [14, ],
        'y': mlt_data,
        'error': mlt_error
    },
    'random': {
        'x': [0, ],
        'y': rnd_data,
        'error': rnd_error
    },
}

add_plots = [
    lambda: plt.hlines(pret_score, 0, 15, linestyles='dashed', label='Pre-trained'),
    ]
scatter_errorbar(data_dict,
                 x_label='Number of output units',
                 y_label='Mean brain score \n (V1, V2, V4, IT, Behavior)',
                 additional_plots=add_plots,
                 folder_name='0821_analysis_tdw_small',
                 fig_name='brainscore_vs_output_num_all',
                 )


# compared with 0705, the difference is (1) change image save folder to 0821_analysis_tdw_small, (2) mlt_data: multi_task_wo_object_class is now multi_task
region_list = ['V1', 'V2', 'V4', 'IT', 'Behavior']
for region in region_list:
    df_mt_r = df_mt[df_mt['benchmark_region'] == region]
    latent_data = list(df_mt_r.groupby('exp_group').mean()['score'].reindex(latent_task_list))
    latent_error = list(df_mt_r.groupby('exp_group').std(ddof=0)['score'].reindex(latent_task_list))

    df_cat_r = df_cat[df_cat['benchmark_region'] == region]
    cat_data = list(df_cat_r.groupby('exp_group').mean()['score'].reindex(cat_task_list))
    cat_error = list(df_cat_r.groupby('exp_group').std(ddof=0)['score'].reindex(cat_task_list))

    mlt_data = df_mt_r.groupby('exp_group').mean()['score']['multi_task']
    mlt_data = [mlt_data, ]
    mlt_error = df_mt_r.groupby('exp_group').std(ddof=0)['score']['multi_task']
    mlt_error = [mlt_error, ]

    # here use multi_task to index group name because it is the default group name
    # the models are random untrained models
    df_rnd_r = df_rnd[df_rnd['benchmark_region'] == region]
    rnd_data = df_rnd_r.groupby('exp_group').mean()['score']['multi_task']
    rnd_data = [rnd_data, ]
    rnd_error = df_rnd_r.groupby('exp_group').std(ddof=0)['score']['multi_task']
    rnd_error = [rnd_error, ]

    pret_score = df_mt_r.groupby('exp_group').mean()['score']['Pre-trained']

    data_dict = {
        'latent_tasks': {
            'x': latent_output_num_list,
            'y': latent_data,
            'error': latent_error
        },
        'cat_tasks': {
            'x': cat_output_num_list,
            'y': cat_data,
            'error': cat_error
        },
        'multi_task': {
            'x': [14, ],
            'y': mlt_data,
            'error': mlt_error
        },
        'random': {
            'x': [0, ],
            'y': rnd_data,
            'error': rnd_error
        },
    }

    add_plots = [
        lambda: plt.hlines(pret_score, 0, 15, linestyles='dashed', label='Pre-trained'),
        ]
    scatter_errorbar(data_dict, 
                     x_label='Number of output units', 
                     y_label=f'{region} score', 
                     additional_plots=add_plots,
                     folder_name='0821_analysis_tdw_small',
                     fig_name=f'brainscore_vs_output_num_{region}',
                    )
