import os
from collections import defaultdict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from config_global import EXP_DIR
from analysis import bar_2par, heatmap, annotate_heatmap
from exp_config_list import multi_task_0620
from activity import get_model_activations_on_dataset, get_neural_activations_on_dataset, target_direction_vector
from utils import prepare_pytorch_model, find_region_layer, get_model_id, abs_cosine_sim, trim_diagonal

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
assert config_list[40]['group_name'] == 'multi_task_wo_object_class'

exp_name = 'multi_task_0620'
results_df = pd.read_csv(os.path.join(EXP_DIR, exp_name,
                                      'brainscore_results.csv'), index_col=0)

model_path_dict = {
    'Pre-trained': None,
    'Categorization': 35,
    'Multi-task': 40,
}

diff_model_actications = defaultdict(dict)
for model_group, run_id in model_path_dict.items():
    if run_id is None:
        load_path = ''
        m_id = 'mt0527-resnet18-pret'
    else:
        config = config_list[run_id]
        load_path = os.path.join(config['save_path'], 'model.pth')
        m_id = get_model_id(config)

    model = prepare_pytorch_model(load_path)

    V4_layer = find_region_layer(results_df, 'V4', m_id)
    IT_layer = find_region_layer(results_df, 'IT', m_id)
    
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
region_list = ['V4_layer', 'IT_layer']

directions_dict = defaultdict(dict)
for group_name, data_dict in diff_model_actications.items():
    layer_list = [data_dict[s] for s in region_list]
    for region, layer in zip(region_list, layer_list):
        directions_dict[group_name][region] = {}
        layer_activity = data_dict['all_activations'][layer]
        for target_name in target_list:
            targets = data_dict['df'][target_name].to_numpy(copy=True)
            directions_dict[group_name][region][target_name] = target_direction_vector(layer_activity, targets)


sim_matrix_dict = defaultdict(dict)
for group_name in group_name_list:
    for region in region_list:
        sim_matrix = np.zeros((len(target_list), len(target_list)))
        for i, target_name_i in enumerate(target_list):
            for j, target_name_j in enumerate(target_list):
                sim_matrix[i, j] = abs_cosine_sim(directions_dict[group_name][region][target_name_i],
                                                  directions_dict[group_name][region][target_name_j])
        sim_matrix_dict[group_name][region] = sim_matrix


for group_name in group_name_list:
    for region in region_list:
        fig, ax = plt.subplots()
        im, cbar = heatmap(sim_matrix_dict[group_name][region],
                           target_list, target_list, ax=ax,
                           cmap='Oranges', cbarlabel="Abs. Cosine Similarity",
                           vmin=0.0, vmax=1.0
                           )
        texts = annotate_heatmap(im, valfmt="{x:.2f}")
        ax.set_title(f'{group_name}: {region}')
        fig.tight_layout()
        plt.show()


diff_targets_results = {}
diff_targets_errors = {}
for target_name in target_list:
    diff_targets_results[target_name] = defaultdict(list)
    diff_targets_errors[target_name] = defaultdict(list)

for group_name in group_name_list:
    for region in region_list:
        sim_mat_ddiag = trim_diagonal(sim_matrix_dict[group_name][region])
        sim_mean = np.mean(sim_mat_ddiag, axis=1)
        sim_std = np.std(sim_mat_ddiag, axis=1)
        
        for i_t, target_name in enumerate(target_list):
            diff_targets_results[target_name][group_name].append(sim_mean[i_t])
            diff_targets_errors[target_name][group_name].append(sim_std[i_t])


for target_name in target_list:
    bar_2par(diff_targets_results[target_name], 
             ['V4_layer', 'IT_layer'], 
             group_name_list, 
             diff_targets_errors[target_name],
             folder_name='0707_analysis',
             fig_name=f'factorization_analysis_{target_name}',
             legend_title=None,
             y_label='Factorization',
             fig_title=f'Factorization on: {target_name}',
             # ylim=[0.0, 1.0],
             )
