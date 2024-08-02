import os
import pandas as pd

from matplotlib import pyplot as plt

from config_global import EXP_DIR
from analysis import bar_2par, adjust_figure, scatter_errorbar
from exp_config_list import multi_task_0620

config_list = multi_task_0620()
df = pd.read_csv(os.path.join(EXP_DIR, 'multi_task_0620', 'brainscore_results.csv'), index_col=0)
df2 = pd.read_csv(os.path.join(EXP_DIR, 'cat_diff_0623', 'brainscore_results.csv'), index_col=0)

latent_task_list = ['size_reg', # 1
                   'translation_reg', # 2
                   'rotation_reg', # 3
                   'size_translation', # 3
                   'size_rotation', # 4
                   'translation_rotation', # 5
                   'size_translation_rotation', # 6
                   # 'categorization', # 8
                   'multi_task_wo_object_class', # 14
                   # 'multi_task' # 78
                   ]
latent_output_num_list = [1, 2, 3, 3, 4, 5, 6, 14]

cat_task_list = ['cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8']
cat_output_num_list = [2, 3, 4, 5, 6, 7, 8]

# the data vary in 3 different groups, 'model', 'exp_group', and 'benchmark_region'
# groupby will collapes the dimensions not specified in groupby
# first collapse 'benchmark_region', and then 'model'
# df.groupby('exp_group').std() will calculate std over both 'model' and 'benchmark_region'
# the following only calculate std over 'model'
latent_data = list(df.groupby(['exp_group', 'model']).mean().groupby('exp_group').mean()['score'].reindex(latent_task_list))
latent_error = list(df.groupby(['exp_group', 'model']).mean().groupby('exp_group').std(ddof=0)['score'].reindex(latent_task_list))

cat_data = list(df2.groupby(['exp_group', 'model']).mean().groupby('exp_group').mean()['score'].reindex(cat_task_list))
cat_error = list(df2.groupby(['exp_group', 'model']).mean().groupby('exp_group').std(ddof=0)['score'].reindex(cat_task_list))

pret_score = df.groupby('exp_group').mean()['score']['Pre-trained']

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
}

add_plots = [
    lambda: plt.hlines(pret_score, 0, 15, linestyles='dashed', label='Pre-trained'),
    ]
scatter_errorbar(data_dict,
                 x_label='Number of output units',
                 y_label='Mean brain score \n (V1, V2, V4, IT, Behavior)',
                 additional_plots=add_plots,
                 folder_name='0620_0623_analysis',
                 fig_name='brainscore_vs_output_num_all',
                 )

region_list = ['V1', 'V2', 'V4', 'IT', 'Behavior']
for region in region_list:
    region_df = df[df['benchmark_region'] == region]
    latent_data = list(region_df.groupby('exp_group').mean()['score'].reindex(latent_task_list))
    latent_error = list(region_df.groupby('exp_group').std(ddof=0)['score'].reindex(latent_task_list))

    region_df2 = df2[df2['benchmark_region'] == region]
    cat_data = list(region_df2.groupby('exp_group').mean()['score'].reindex(cat_task_list))
    cat_error = list(region_df2.groupby('exp_group').std(ddof=0)['score'].reindex(cat_task_list))

    pret_score = region_df.groupby('exp_group').mean()['score']['Pre-trained']

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
    }

    add_plots = [
        lambda: plt.hlines(pret_score, 0, 15, linestyles='dashed', label='Pre-trained'),
        ]
    scatter_errorbar(data_dict, 
                     x_label='Number of output units', 
                     y_label=f'{region} score', 
                     additional_plots=add_plots,
                     folder_name='0620_0623_analysis',
                     fig_name=f'brainscore_vs_output_num_{region}',
                    )
