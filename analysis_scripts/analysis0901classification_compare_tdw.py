import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from matplotlib import pyplot as plt

from utils import prepare_pytorch_model
from config_global import DEVICE, EXP_DIR, FIG_DIR
from analysis import bar_2par, adjust_figure, scatter_errorbar
from exp_config_list import multi_task_tdw_multiscene_0826, cat_diff_tdw_multiscene_0826
from activity import get_model_activations_on_dataset, get_neural_activations_on_dataset, cross_validate_on_target

region_list = ['V4', 'IT']
neural_activations, neural_df= get_neural_activations_on_dataset(region_list)

# distance_reg: 0-4
# translation_reg: 5-9
# rotation_reg: 10-14
# distance_translation: 15-19
# distance_rotation: 20-24
# translation_rotation: 25-29
# distance_translation_rotation: 30-34
# categorization: 35-39
# multi_task: 40-44 (without object categorization)
config_list = multi_task_tdw_multiscene_0826()
assert config_list[0]['group_name'] == 'distance_reg'
assert config_list[5]['group_name'] == 'translation_reg'
assert config_list[10]['group_name'] == 'rotation_reg'
assert config_list[15]['group_name'] == 'distance_translation'
assert config_list[20]['group_name'] == 'distance_rotation'
assert config_list[25]['group_name'] == 'translation_rotation'
assert config_list[30]['group_name'] == 'distance_translation_rotation'
assert config_list[35]['group_name'] == 'categorization'
assert config_list[40]['group_name'] == 'multi_task'

run_id_dict = {
    'pre-trained': -1,
    'distance_reg': 0,
    'translation_reg': 5,
    'rotation_reg': 10,
    'distance_translation': 15,
    'distance_rotation': 20,
    'translation_rotation': 25,
    'distance_translation_rotation': 30,
    'categorization': 35,
    'multi_task': 40, # w/o object class
}

diff_model_actications = defaultdict(dict)
for model_group, run_id in run_id_dict.items():
    if run_id == -1:
        load_path = ''
    else:
        load_path = os.path.join(EXP_DIR, 'multi_task_tdw_multiscene_0826', f'run_{run_id:04d}', 'model.pth')
    model = prepare_pytorch_model(load_path=load_path)
    layer_list = ['avgpool']
    all_activations, df= get_model_activations_on_dataset(model, layer_list)
    diff_model_actications[model_group]['all_activations'] = all_activations
    diff_model_actications[model_group]['df'] = df

config_list = cat_diff_tdw_multiscene_0826()
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

for model_group, run_id in cat_run_id_dict.items():
    load_path = os.path.join(EXP_DIR, 'cat_diff_tdw_multiscene_0826', f'run_{run_id:04d}', 'model.pth')
    model = prepare_pytorch_model(load_path=load_path)
    layer_list = ['avgpool']
    all_activations, df= get_model_activations_on_dataset(model, layer_list)
    diff_model_actications[model_group]['all_activations'] = all_activations
    diff_model_actications[model_group]['df'] = df

diff_model_actications['Neural-IT'] = {
    'all_activations': neural_activations,
    'df': neural_df,
}

target_name = 'category_label'

category_results = {}
category_error = {}
for group_name, data_dict in diff_model_actications.items():
    if group_name == 'Neural-IT':
        layer = 'IT'
    else:
        layer = 'avgpool'
    layer_activity = data_dict['all_activations'][layer]
    acc_mean, acc_std = cross_validate_on_target(layer_activity,
                                                 data_dict['df'],
                                                 target_name,
                                                 num_cross_val=30,
                                                 mode='classification')


    category_results[group_name] = acc_mean
    category_error[group_name] = acc_std


latent_task_list = ['distance_reg', # 1
                   'translation_reg', # 2
                   'rotation_reg', # 3
                   'distance_translation', # 3
                   'distance_rotation', # 4
                   'translation_rotation', # 5
                   'distance_translation_rotation', # 6
                   # 'categorization', # 8
                   'multi_task', # 14
                   ]
latent_output_num_list = [1, 2, 3, 3, 4, 5, 6, 14]

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
    # 'neural_IT': {
    #     'x': [15, ],
    #     'y': [category_results['Neural-IT'], ],
    #     'error': [category_error['Neural-IT'], ]
    # },
}

pret_score = category_results['pre-trained']
pret_error = category_error['pre-trained']

IT_score = category_results['Neural-IT']
IT_error = category_error['Neural-IT']

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
                 folder_name='0830_analysis_tdw_small_multi_scene',
                 fig_name='category_acc_vs_output_num_all',
                 )
