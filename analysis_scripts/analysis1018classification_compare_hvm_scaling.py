# %%
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from matplotlib import pyplot as plt

from config_global import DEVICE, EXP_DIR, FIG_DIR
from analysis import bar_2par, adjust_figure, scatter_errorbar
from exp_config_list import multi_task_0620, cat_diff_0623
from activity import get_model_activations_on_dataset, get_neural_activations_on_dataset, cross_validate_on_target

# %%
region_list = ['V4', 'IT']
neural_activations, neural_df= get_neural_activations_on_dataset(region_list)

# %%
config_list = multi_task_0620()
assert config_list[0]['group_name'] == 'size_reg'
assert config_list[5]['group_name'] == 'translation_reg'
assert config_list[10]['group_name'] == 'rotation_reg'
assert config_list[15]['group_name'] == 'size_translation'
assert config_list[20]['group_name'] == 'size_rotation'
assert config_list[25]['group_name'] == 'translation_rotation'
assert config_list[30]['group_name'] == 'size_translation_rotation'
assert config_list[35]['group_name'] == 'categorization'
assert config_list[40]['group_name'] == 'multi_task_wo_object_class'

# %%
fig_folder_name = '1018_analysis_decoding_scaling_hvm'
exp_name = 'multi_task_0620'
score_df = pd.read_csv(os.path.join(EXP_DIR, exp_name, 'brainscore_results.csv'), index_col=0)

# %%
run_id_dict = {
    'Pre-trained': 'pret',
    'size_reg': 0,
    'translation_reg': 5,
    'rotation_reg': 10,
    'size_translation': 15,
    'size_rotation': 20,
    'translation_rotation': 25,
    'size_translation_rotation': 30,
    'categorization': 35,
    'Multi-task': 40, # w/o object class
}

# %%
model_ITlayer_dict = {}
for model_group, run_id in run_id_dict.items():
    if run_id == 'pret':
        model_id = 'mt0527-resnet18-pret'
    else:
        model_id = f'{exp_name}-resnet18-{run_id}'
    layer_list = list(score_df[(score_df['model'] == model_id) & (score_df['benchmark_region'] == 'IT')]['mapped_layer'])
    assert len(layer_list) == 1
    model_ITlayer_dict[model_group] = layer_list[0]

# %%
model_ITlayer_dict

# %%
from utils import prepare_pytorch_model


# %%
# TODO: can be speeded up by using a GPU
diff_model_actications = defaultdict(dict)
for model_group, run_id in run_id_dict.items():
    print(f'Processing {model_group}...')
    if run_id == 'pret':
        load_path = ''
    else:
        load_path = os.path.join(EXP_DIR, exp_name, f'run_{run_id:04d}', 'model.pth')
    model = prepare_pytorch_model(out_dim=78, load_path=load_path)
    
    # need to get brainscore assigned IT layer instead of the avgpool layer
    all_activations, df = get_model_activations_on_dataset(model, [model_ITlayer_dict[model_group]])
    diff_model_actications[model_group]['all_activations'] = all_activations
    diff_model_actications[model_group]['df'] = df

# %%
diff_model_actications['Neural-IT'] = {
    'all_activations': neural_activations,
    'df': neural_df,
}
model_ITlayer_dict['Neural-IT'] = 'IT'

# %%
config_list = cat_diff_0623()
assert config_list[0]['group_name'] == 'cat2'
assert config_list[5]['group_name'] == 'cat3'
assert config_list[10]['group_name'] == 'cat4'
assert config_list[15]['group_name'] == 'cat5'
assert config_list[20]['group_name'] == 'cat6'
assert config_list[25]['group_name'] == 'cat7'
assert config_list[30]['group_name'] == 'cat8'

cat_run_id_dict = {
    'cat2': 0,
    'cat3': 5,
    'cat4': 10,
    'cat5': 15,
    'cat6': 20,
    'cat7': 25,
    'cat8': 30,
}

# %%
cat_exp_name = 'cat_diff_0623'
cat_score_df = pd.read_csv(os.path.join(EXP_DIR, cat_exp_name, 'brainscore_results.csv'), index_col=0)

# %%
for model_group, run_id in cat_run_id_dict.items():
    if run_id == 'pret':
        model_id = 'mt0527-resnet18-pret'
    else:
        model_id = f'{cat_exp_name}-resnet18-{run_id}'
    layer_list = list(cat_score_df[(cat_score_df['model'] == model_id) & (cat_score_df['benchmark_region'] == 'IT')]['mapped_layer'])
    assert len(layer_list) == 1
    model_ITlayer_dict[model_group] = layer_list[0]

# %%
model_ITlayer_dict

# %%
for model_group, run_id in cat_run_id_dict.items():
    print(f'Processing {model_group}...')
    if run_id == 'pret':
        load_path = ''
    else:
        load_path = os.path.join(EXP_DIR, cat_exp_name, f'run_{run_id:04d}', 'model.pth')
    model = prepare_pytorch_model(out_dim=78, load_path=load_path)
    
    # need to get brainscore assigned IT layer instead of the avgpool layer
    all_activations, df = get_model_activations_on_dataset(model, [model_ITlayer_dict[model_group]])
    diff_model_actications[model_group]['all_activations'] = all_activations
    diff_model_actications[model_group]['df'] = df

# %%
target_name = 'category_label'

category_results = {}
category_error = {}
for group_name, data_dict in diff_model_actications.items():
    print(f'Processing {group_name}...')
    layer_activity = data_dict['all_activations'][model_ITlayer_dict[group_name]]
    acc_mean, acc_std = cross_validate_on_target(layer_activity,
                                                 data_dict['df'],
                                                 target_name,
                                                 num_cross_val=10,
                                                 mode='classification')
    category_results[group_name] = acc_mean
    category_error[group_name] = acc_std

# %%
latent_task_list = ['size_reg', # 1
                   'translation_reg', # 2
                   'rotation_reg', # 3
                   'size_translation', # 3
                   'size_rotation', # 4
                   'translation_rotation', # 5
                   'size_translation_rotation', # 6
                   # 'categorization', # 8
                   # 'Multi-task', # 14
                   ]
latent_output_num_list = [1, 2, 3, 3, 4, 5, 6]

latent_data = []
latent_error = []
for task in latent_task_list:
    latent_data.append(category_results[task])
    latent_error.append(category_error[task])

cat_task_list = ['cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8']
cat_output_num_list = [2, 3, 4, 5, 6, 7, 8]

cat_data = []
cat_error = []
for task in cat_task_list:
    cat_data.append(category_results[task])
    cat_error.append(category_error[task])


plot_data_dict = {
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
    'multi-task': {
        'x': [14, ],
        'y': [category_results['Multi-task'], ],
        'error': [category_error['Multi-task'], ]
    },
}

pret_score = category_results['Pre-trained']
pret_error = category_error['Pre-trained']

IT_score = category_results['Neural-IT']
IT_error = category_error['Neural-IT']

# %%
add_plots = [
    lambda: plt.hlines(pret_score, 0, 15, linestyles='dashed', label='Pre-trained', color='C4'),
    lambda: plt.fill_between([0, 15], 2 * [pret_score - pret_error], 2 * [pret_score + pret_error], alpha=0.2, color='C4'),
    lambda: plt.hlines(IT_score, 0, 15, linestyles='dashed', label='neural-IT', color='C3'),
    lambda: plt.fill_between([0, 15], 2 * [IT_score - IT_error], 2 * [IT_score + IT_error], alpha=0.2, color='C3'),
    ]
scatter_errorbar(plot_data_dict,
                 x_label='Number of output units',
                 y_label='Category decoding accuracy',
                 additional_plots=add_plots,
                 folder_name=fig_folder_name,
                 fig_name='category_acc_vs_output_num_all',
                 )


# %% [markdown]
# ### Scaling analysis

# %%
from tqdm import tqdm

# %%
diff_model_actications.keys()

# %%
target_name = 'category_label'
group_name_list = ['Pre-trained', 'size_translation_rotation', 'categorization', 'Multi-task', 'Neural-IT']
cv_unit_number = {}
cv_results = {}
cv_errors = {}
for group_name in group_name_list:
    print(f'Processing {group_name}...')
    data_dict = diff_model_actications[group_name]
    layer_activity = data_dict['all_activations'][model_ITlayer_dict[group_name]]
    downsample_number_list = [3 ** i for i in range(1, 20) if 3 ** i <= layer_activity.shape[1]]
    # for neural data, have the largest population
    if downsample_number_list[-1] <= 200:
        downsample_number_list.append(layer_activity.shape[1])
    cv_unit_number[group_name] = downsample_number_list
    
    cv_results[group_name] = []
    cv_errors[group_name] = []
    for downsample_number in tqdm(downsample_number_list):
        assert downsample_number <= layer_activity.shape[1]
        acc_mean, acc_std = cross_validate_on_target(layer_activity,
                                                        data_dict['df'],
                                                        target_name,
                                                        num_cross_val=5,
                                                        downsample_number=downsample_number,
                                                        mode='classification',
                                                        )
        cv_results[group_name].append(acc_mean)
        cv_errors[group_name].append(acc_std)

# %%
fig, ax = plt.subplots()
for group_name in group_name_list:
    ax.errorbar(cv_unit_number[group_name], cv_results[group_name], yerr=cv_errors[group_name], 
                fmt='o-', label=group_name)
ax.set_xscale('log')
ax.set_xlabel('Number of units')
ax.set_ylabel('Category decoding accuracy')
ax.legend()
ax.set_ylim(0.0, 1.0)
adjust_figure(ax)
if not os.path.exists(os.path.join(FIG_DIR, fig_folder_name)):
    os.makedirs(os.path.join(FIG_DIR, fig_folder_name))
fig.savefig(os.path.join(FIG_DIR, fig_folder_name, 'category_decoding_scaling' + '.pdf'), transparent=True)

# %%



