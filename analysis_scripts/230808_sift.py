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
from utils import prepare_pytorch_model
from sift import get_sift_features

region_list = ['V4', 'IT']
neural_activations, neural_df= get_neural_activations_on_dataset(region_list)

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

model_id_dict = {
    'Pre-trained': None,
    'categorization': 35,
}

diff_model_actications = defaultdict(dict)
for model_group, model_id in model_id_dict.items():
    if model_id is not None:
        load_path = os.path.join(EXP_DIR, 'multi_task_0620', f'run_{model_id:04d}', 'model.pth')
    else:
        load_path = ''
    model = prepare_pytorch_model(load_path)
    layer_list = ['avgpool']
    all_activations, df = get_model_activations_on_dataset(model, layer_list)
    diff_model_actications[model_group]['all_activations'] = all_activations
    diff_model_actications[model_group]['df'] = df

diff_model_actications['Neural-IT'] = {
    'all_activations': neural_activations,
    'df': neural_df,
}

sift_activateions, sift_df = get_sift_features(num_kps=5)

diff_model_actications['SIFT'] = {
    'all_activations': sift_activateions,
    'df': sift_df,
}

target_name = 'category_label'

category_results = {}
category_error = {}
for group_name, data_dict in diff_model_actications.items():
    if group_name == 'Neural-IT':
        layer = 'IT'
    elif group_name == 'SIFT':
        layer = 'SIFT'
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

print(category_results)
print(category_error)

group_names = ['Pre-trained', 'categorization', 'Neural-IT', 'SIFT']
bar_2par(category_results, 
         ['Classification Decoding Accuracy'], 
         group_names, 
         data_err=category_error,
         folder_name='analysis0808sift',
         fig_name='category_accuracy')
