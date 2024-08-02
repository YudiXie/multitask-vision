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


target_list = ['s', 'ty', 'tz', 'rxy_semantic', 'rxz_semantic', 'ryz_semantic']
group_name_list = ['Pre-trained', 'Categorization', 'Multi-task', 'Neural-data']

diff_target_results = defaultdict(dict)
diff_target_error = defaultdict(dict)
for target_name in target_list:
    for group_name, data_dict in diff_model_actications.items():
        mean_list = []
        std_list = []
        layer_list = [data_dict['V4_layer'], data_dict['IT_layer']]
        for layer in layer_list:
            layer_activity = data_dict['all_activations'][layer]
            coef_mean, coef_std = cross_validate_on_target(layer_activity, data_dict['df'], target_name)
            mean_list.append(coef_mean)
            std_list.append(coef_std)

        diff_target_results[target_name][group_name] = mean_list
        diff_target_error[target_name][group_name] = std_list


for target_name in target_list:
    bar_2par(diff_target_results[target_name], 
             ['V4_layer', 'IT_layer'], 
             group_name_list, 
             diff_target_error[target_name],
             folder_name='0620_0623_analysis',
             fig_name=f'decoding_regression_model_vs_neural_{target_name}',
             legend_title=None,
             y_label='Pearson correlation coefficient',
             fig_title=f'Performance on: {target_name}',
             # ylim=[0.0, 1.0],
             )


aggregate_results = defaultdict(list)
for target_name in target_list:
    for group_name in group_name_list:
        aggregate_results[group_name] += diff_target_results[target_name][group_name]


rmsd_dict = defaultdict(list)
for group_name in group_name_list:
    if group_name == 'Neural-data':
        continue
    predict = np.array(aggregate_results[group_name])
    target = np.array(aggregate_results['Neural-data'])
    plt.scatter(target, predict, label=group_name)
    rmsd = np.sqrt(np.square(predict - target).mean())
    coef, _ignore = stats.pearsonr(predict, target)
    rmsd_dict[group_name].append(rmsd)
    print(f'Group name: {group_name}, RMSD: {rmsd}')
    print(f'Group name: {group_name}, Pearson coef: {coef}')
plt.plot([0.0, 1.0], [0.0, 1.0], 'r--')
plt.legend()
plt.xlabel('Neural decoding performance')
plt.ylabel('Model decoding performance')
plt.xlim([0.0, 0.8])
plt.ylim([0.0, 0.8])
plt.gca().set_aspect('equal', 'box')
adjust_figure()
plt.savefig(os.path.join(FIG_DIR, '0620_0623_analysis', 'all_performance_model_vs_neural' + '.pdf'), transparent=True)
plt.show()
plt.close()


bar_2par(rmsd_dict,
         ['Pre-trained   Categorization   Multi-task'], 
         ['Pre-trained', 'Categorization', 'Multi-task'],
         y_label='RMSD',
         fig_title='RMSD between model and neural decoding performance',
         fig_name='RMSD_between_model_and_neural_decoding',
         folder_name='0620_0623_analysis',
         legend_title=None,
         xlim=[-0.7, 1.0],
         bar_label_decimals=3,
         )


# manually assigned layers analysis
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


target_list = ['s', 'ty', 'tz', 'rxy_semantic', 'rxz_semantic', 'ryz_semantic']
group_name_list = ['Pre-trained', 'Categorization', 'Multi-task', 'Neural-data']

diff_target_results_manual_layer = defaultdict(dict)
diff_target_error_manual_layer = defaultdict(dict)
for target_name in target_list:
    for group_name, data_dict in diff_model_actications_manual_layer.items():
        mean_list = []
        std_list = []
        layer_list = [data_dict['V4_layer'], data_dict['IT_layer']]
        for layer in layer_list:
            layer_activity = data_dict['all_activations'][layer]
            coef_mean, coef_std = cross_validate_on_target(layer_activity, data_dict['df'], target_name)
            mean_list.append(coef_mean)
            std_list.append(coef_std)

        diff_target_results_manual_layer[target_name][group_name] = mean_list
        diff_target_error_manual_layer[target_name][group_name] = std_list


for target_name in target_list:
    bar_2par(diff_target_results_manual_layer[target_name], 
             ['V4_layer3.1.relu', 'IT_avgpool'], 
             group_name_list, 
             diff_target_error_manual_layer[target_name],
             folder_name='0620_0623_analysis',
             fig_name=f'decoding_regression_model_vs_neural_{target_name}_manual_layer',
             legend_title=None,
             y_label='Pearson correlation coefficient',
             fig_title=f'Performance on: {target_name} manual layer',
             # ylim=[0.0, 1.0],
             )


aggregate_results_manual_layer = defaultdict(list)
for target_name in target_list:
    for group_name in group_name_list:
        aggregate_results_manual_layer[group_name] += diff_target_results_manual_layer[target_name][group_name]


rmsd_dict_manual_layer = defaultdict(list)
for group_name in group_name_list:
    if group_name == 'Neural-data':
        continue
    predict = np.array(aggregate_results_manual_layer[group_name])
    target = np.array(aggregate_results_manual_layer['Neural-data'])
    plt.scatter(target, predict, label=group_name)
    rmsd = np.sqrt(np.square(predict - target).mean())
    coef, _ignore = stats.pearsonr(predict, target)
    rmsd_dict_manual_layer[group_name].append(rmsd)
    print(f'Group name: {group_name}, RMSD: {rmsd}')
    print(f'Group name: {group_name}, Pearson coef: {coef}')
plt.plot([0.0, 1.0], [0.0, 1.0], 'r--')
plt.legend()
plt.xlabel('Neural decoding performance')
plt.ylabel('Model decoding performance (manual layer)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.gca().set_aspect('equal', 'box')
adjust_figure()
plt.savefig(os.path.join(FIG_DIR, '0620_0623_analysis', 'all_performance_model_vs_neural_manual_layer' + '.pdf'), transparent=True)
plt.show()
plt.close()


bar_2par(rmsd_dict_manual_layer,
         ['Pre-trained   Categorization   Multi-task'], 
         ['Pre-trained', 'Categorization', 'Multi-task'],
         y_label='RMSD',
         fig_title='RMSD between model and neural decoding performance (manual layer)',
         fig_name='RMSD_between_model_and_neural_decoding_manual_layer',
         folder_name='0620_0623_analysis',
         legend_title=None,
         xlim=[-0.7, 1.0],
         bar_label_decimals=3,
         )

