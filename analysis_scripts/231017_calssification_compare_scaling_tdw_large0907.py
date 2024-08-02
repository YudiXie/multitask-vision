# %%
import os
from collections import defaultdict
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import prepare_pytorch_model
from config_global import DEVICE, EXP_DIR, FIG_DIR
from analysis import adjust_figure, scatter_errorbar
from activity import get_model_activations_on_dataset, get_neural_activations_on_dataset, cross_validate_on_target

from exp_config_list import multi_task_tdw_large20230907_nopret_0925

# %%
region_list = ['V4', 'IT']
neural_activations, neural_df = get_neural_activations_on_dataset(region_list)

# %%
config_list = multi_task_tdw_large20230907_nopret_0925()
assert config_list[0]['group_name'] == 'distance_reg'
assert config_list[3]['group_name'] == 'translation_reg'
assert config_list[6]['group_name'] == 'rotation_reg'
assert config_list[9]['group_name'] == 'distance_translation'
assert config_list[12]['group_name'] == 'distance_rotation'
assert config_list[15]['group_name'] == 'translation_rotation'
assert config_list[18]['group_name'] == 'distance_translation_rotation'
assert config_list[21]['group_name'] == 'category_class'
assert config_list[24]['group_name'] == 'object_class'
assert config_list[27]['group_name'] == 'cat_obj_class_all_latents'

# %%
fig_folder_name = '1017_analysis_decoding_scaling_tdw_large0907'
exp_name = 'multi_task_tdw_large20230907_nopret_0925'
score_df = pd.read_csv(os.path.join(EXP_DIR, exp_name, 'brainscore_results.csv'), index_col=0)

# %%
run_id_dict = {
    'pre-trained': 'pret',
    'distance_reg': 0,
    'translation_reg': 3,
    'rotation_reg': 6,
    'distance_translation': 9,
    'distance_rotation': 12,
    'translation_rotation': 15,
    'distance_translation_rotation': 18,
    'category_class': 21,
    'object_class': 24,
    'cat_obj_class_all_latents': 27,
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
# TODO: can be speeded up by using a GPU
diff_model_actications = defaultdict(dict)
for model_group, run_id in run_id_dict.items():
    print(f'Processing {model_group}...')
    if run_id == 'pret':
        load_path = ''
    else:
        load_path = os.path.join(EXP_DIR, exp_name, f'run_{run_id:04d}', 'model.pth')
    model = prepare_pytorch_model(out_dim=710, load_path=load_path)
    
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
target_name = 'category_label'

category_results = {}
category_error = {}
for group_name, data_dict in diff_model_actications.items():
    print(f'Processing {group_name}...')
    layer_activity = data_dict['all_activations'][model_ITlayer_dict[group_name]]
    acc_mean, acc_std = cross_validate_on_target(layer_activity,
                                                 data_dict['df'],
                                                 target_name,
                                                 num_cross_val=30,
                                                 mode='classification')
    category_results[group_name] = acc_mean
    category_error[group_name] = acc_std

# %%
latent_task_list = ['distance_reg', # 1
                    'translation_reg', # 2
                    'rotation_reg', # 3
                    'distance_translation', # 3
                    'distance_rotation', # 4
                    'translation_rotation', # 5
                    'distance_translation_rotation', # 6
                    ]
latent_output_num_list = [1, 2, 3, 3, 4, 5, 6]

latent_data = []
latent_error = []
for task in latent_task_list:
    latent_data.append(category_results[task])
    latent_error.append(category_error[task])


plot_data_dict = {
    'latent_tasks': {
        'x': latent_output_num_list,
        'y': latent_data,
        'error': latent_error
    },
    'multi-task': {
        'x': [0, ],
        'y': [category_results['cat_obj_class_all_latents'], ],
        'error': [category_error['cat_obj_class_all_latents'], ]
    },
    'category_class': {
        'x': [0, ],
        'y': [category_results['category_class'], ],
        'error': [category_error['category_class'], ],
    },
    'object_class': {
        'x': [0, ],
        'y': [category_results['object_class'], ],
        'error': [category_error['object_class'], ],
    },
}

pret_score = category_results['pre-trained']
pret_error = category_error['pre-trained']

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

# %%
target_name = 'category_label'
group_name_list = ['pre-trained', 'distance_translation_rotation', 'category_class', 'Neural-IT']
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



