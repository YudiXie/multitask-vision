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
from analysis import bar_2par, adjust_figure
from exp_config_list import multi_task_0620
from activity import get_model_activations_on_dataset, get_neural_activations_on_dataset, cross_validate_on_target

region_list = ['V4', 'IT']
neural_activations, neural_df= get_neural_activations_on_dataset(region_list)


def prepare_model(run_id=None):
    exp_name = 'multi_task_0620'

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 78)
    model = model.to(DEVICE)

    if run_id is not None:
        load_path = os.path.join(EXP_DIR, f'{exp_name}', f'run_{run_id:04d}', 'model.pth')
        print(f'loading model from {load_path}')
        model.load_state_dict(torch.load(load_path, map_location=DEVICE), strict=True)
    return model

# size_reg: 0-4
# translation_reg: 5-9
# rotation_reg: 10-14
# size_translation: 15-19
# size_rotation: 20-24
# translation_rotation: 25-29
# size_translation_rotation: 30-34
# categorization: 35-39
# multi_task_wo_object_class: 40-44
# multi_task: 45-49
config_list = multi_task_0620()
assert config_list[35]['group_name'] == 'categorization'
assert config_list[40]['group_name'] == 'multi_task_wo_object_class'


results_df = pd.read_csv(os.path.join(EXP_DIR, 'multi_task_0620', 'brainscore_results.csv'), index_col=0)

def find_region_layer(region, model_n):
    layer_series = results_df[(results_df['model'] == model_n) & (results_df['benchmark_region'] == region)]['mapped_layer']
    assert len(layer_series) == 1
    layer_name = layer_series.to_numpy(copy=True)[0]
    return layer_name


model_id_dict = {
    'Pre-trained': None,
    'Categorization': 35,
    'Multi-task': 40, # w/o object class
}

diff_model_actications = defaultdict(dict)
for model_group, model_id in model_id_dict.items():
    model = prepare_model(run_id=model_id)
    if model_id is None:
        # pre-trained model
        m_name = 'mt0527-resnet18-pret'
    else:
        m_name = f'multi_task_0620-resnet18-{model_id}'

    V4_layer = find_region_layer('V4', m_name)
    IT_layer = find_region_layer('IT', m_name)
    
    layer_list = [V4_layer, IT_layer]
    all_activations, df= get_model_activations_on_dataset(model, layer_list)
    diff_model_actications[model_group]['all_activations'] = all_activations
    diff_model_actications[model_group]['df'] = df
    diff_model_actications[model_group]['V4_layer'] = V4_layer
    diff_model_actications[model_group]['IT_layer'] = IT_layer


diff_model_actications['Neural-data'] = {
    'all_activations': neural_activations,
    'df': neural_df,
    'V4_layer': 'V4',
    'IT_layer': 'IT',
}

group_name_list = ['Pre-trained', 'Categorization', 'Multi-task', 'Neural-data']
target_name = 'category_label'

category_results = {}
category_error = {}
for group_name, data_dict in diff_model_actications.items():
    mean_list = []
    std_list = []
    layer_list = [data_dict['V4_layer'], data_dict['IT_layer']]
    for layer in layer_list:
        layer_activity = data_dict['all_activations'][layer]
        acc_mean, acc_std = cross_validate_on_target(layer_activity, 
                                                     data_dict['df'], 
                                                     target_name, 
                                                     num_cross_val=30, 
                                                     mode='classification')
        mean_list.append(acc_mean)
        std_list.append(acc_std)

    category_results[group_name] = mean_list
    category_error[group_name] = std_list


bar_2par(category_results,
         ['V4_layer', 'IT_layer'],
         group_name_list,
         category_error,
         folder_name='0620_0623_analysis',
         fig_name=f'decoding_classification_model_vs_neural_category_label',
         legend_title=None,
         y_label='Classification accuracy',
         fig_title=f'Performance on: category_label',
         # ylim=[0.0, 1.0],
         )

## manually assigned layers analysis
# Repeat the same analysis as before but with manually assigned layers
# V4_layer = 'layer3.1.relu'
# IT_layer = 'avgpool’
# which is different from previous analysis where layers are assigned by brainscore

model_id_dict = {
    'Pre-trained': None,
    'Categorization': 35,
    'Multi-task': 40, # w/o object class
}

diff_model_actications_manual_layer = defaultdict(dict)
for model_group, model_id in model_id_dict.items():
    model = prepare_model(run_id=model_id)
    if model_id is None:
        # pre-trained model
        m_name = 'mt0527-resnet18-pret'
    else:
        m_name = f'multi_task_0620-resnet18-{model_id}'

    V4_layer = 'layer3.1.relu'
    IT_layer = 'avgpool'
    
    layer_list = [V4_layer, IT_layer]
    all_activations, df= get_model_activations_on_dataset(model, layer_list)
    diff_model_actications_manual_layer[model_group]['all_activations'] = all_activations
    diff_model_actications_manual_layer[model_group]['df'] = df
    diff_model_actications_manual_layer[model_group]['V4_layer'] = V4_layer
    diff_model_actications_manual_layer[model_group]['IT_layer'] = IT_layer


diff_model_actications_manual_layer['Neural-data'] = {
    'all_activations': neural_activations,
    'df': neural_df,
    'V4_layer': 'V4',
    'IT_layer': 'IT',
}


group_name_list = ['Pre-trained', 'Categorization', 'Multi-task', 'Neural-data']
target_name = 'category_label'

category_results_manual_layer = {}
category_error_manual_layer = {}
for group_name, data_dict in diff_model_actications_manual_layer.items():
    mean_list = []
    std_list = []
    layer_list = [data_dict['V4_layer'], data_dict['IT_layer']]
    for layer in layer_list:
        layer_activity = data_dict['all_activations'][layer]
        acc_mean, acc_std = cross_validate_on_target(layer_activity, 
                                                     data_dict['df'], 
                                                     target_name, 
                                                     num_cross_val=30, 
                                                     mode='classification')
        mean_list.append(acc_mean)
        std_list.append(acc_std)

    category_results_manual_layer[group_name] = mean_list
    category_error_manual_layer[group_name] = std_list


bar_2par(category_results_manual_layer,
         ['V4_layer3.1.relu', 'IT_avgpool'],
         group_name_list,
         category_error_manual_layer,
         folder_name='0620_0623_analysis',
         fig_name=f'decoding_classification_model_vs_neural_category_label_manual_layer',
         legend_title=None,
         y_label='Classification accuracy',
         fig_title=f'Performance on: category_label manual layer',
         # ylim=[0.0, 1.0],
         )

