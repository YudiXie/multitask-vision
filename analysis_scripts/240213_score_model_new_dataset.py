# %%
import os
import numpy as np
import pandas as pd

import scipy.stats
from matplotlib import pyplot as plt

from config_global import EXP_DIR, FIG_DIR
from analysis import scatter_errorbar, adjust_figure

# %%
# compared with previous, the difference is to change the experiement name
df_mt = pd.read_csv(os.path.join(EXP_DIR, 'multi_task_tdw_1m20240206_nopret_0206', 'brainscore_results.csv'), index_col=0)
df_rnd = pd.read_csv(os.path.join(EXP_DIR, 'random_models0630', 'brainscore_results.csv'), index_col=0)

# %%
# compared with previous, updated rotation output number

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
df_mt['exp_group'].unique()

# %%
# compared with 1110, the difference is 
# (1) change image save folder to 0213_tdw_1m_updated_dataset_nopret
# (2) pret_score read out from df_rnd_neural instead of df_mt_neural
# (3) change the numbers of output units to reflect the new dataset
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

pret_score = df_rnd_neural.groupby('exp_group').mean()['score']['Pre-trained']

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

add_plots = [
    lambda: plt.scatter([1000, ], [pret_score, ], label='ImageNet-1K', color='r', marker='D'),
    lambda: plt.hlines(rnd_data[0], 1, 1000, linestyles='dashed', label='Untrained', color='k'),
    lambda: plt.fill_between([1, 1000], 2 * [rnd_data[0] - rnd_error[0]], 2 * [rnd_data[0] + rnd_error[0]], alpha=0.2, color='k'),
    ]
scatter_errorbar(data_dict,
                 x_label='Number of CNN output units',
                 y_label='Mean Brain-Score \n (V1, V2, V4, IT)',
                 additional_plots=add_plots,
                 folder_name='0213_tdw_1m_updated_dataset_nopret',
                 fig_name='brainscore_vs_output_num_wo_behavior',
                 log_scale=True,
                 )

# %%



