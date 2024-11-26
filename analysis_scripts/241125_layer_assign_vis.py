# %%
import numpy as np
import pandas as pd
from pathlib import Path

from matplotlib import pyplot as plt

from config_global import EXP_DIR, FIG_DIR, DATA_DIR
import easyfigs.basicplot as bp
from utils import get_model_group_dist

# %%
df_mt = pd.read_csv(Path(EXP_DIR).joinpath('multi_task_tdw_1m20240206_0718', 'brainscore_results.csv'), index_col=0)
df_pt_rnd = pd.read_csv(Path(EXP_DIR).joinpath('pretrain_and_random_resnet18_0220', 'brainscore_results.csv'), index_col=0)

df_nc_cat = pd.read_csv(Path(EXP_DIR).joinpath('cat_tdw_1m_nc_20240902_0902', 'brainscore_results.csv'), index_col=0)
nc_num_list = [2, 4, 6, 8, 16]
nc_list = [f'{i}c' for i in nc_num_list]
nc_cat_groups = []
for nc in nc_list:
    nc_cat_groups.extend([nc, ] * 30)
df_nc_cat['exp_group'] = nc_cat_groups

df_imn = pd.read_csv(Path(EXP_DIR).joinpath('imagenet1k_0902', 'brainscore_results.csv'), index_col=0)

# %%
layer2idx = {
    'relu': 1,
    'layer1.0.relu': 3,
    'layer1.1.relu': 5,
    'layer2.0.relu': 7,
    'layer2.1.relu': 9,
    'layer3.0.relu': 11,
    'layer3.1.relu': 13,
    'layer4.0.relu': 15,
    'layer4.1.relu': 17,
    'avgpool': 19,
    'fc': 21,
}
xaxis_layer_list = list(layer2idx.keys())
xaxis_layer_idx_list = list(layer2idx.values())

tasks_dict = {'Distance': [i for i in range(8)],
              'Translation': [i for i in range(8, 16)],
              'Rotation': [i for i in range(16, 24)],
              'Dis. Tra. Rot.': [i for i in range(24, 32)],
              'Obj. Category': [i for i in range(32, 40)],
              'All spatial + cla.': [i for i in range(40, 48)],
              'ImageNet-1K': [i for i in range(48, 56)],
              'Untrained': [i for i in range(56, 61)],
              }
labels = list(tasks_dict.keys())

record_layers = ['layer1.0.relu',
                 'layer1.1.relu',
                 'layer2.0.relu', 
                 'layer2.1.relu',
                 'layer3.0.relu', 
                 'layer3.1.relu',
                 'layer4.0.relu', 
                 'layer4.1.relu',
                 'avgpool',
                 ]
dataset_name = 'tdw_1m_20240206_val_0_04'

# visualize similarity to reference model
ref_group = 'Obj. Category'
ref_group_idx = labels.index(ref_group)

dis2ref_mean_list = []
dis2ref_std_list = []
for i, layer in enumerate(record_layers):
    layer_name = layer.replace('.', '_')
    sim_matrix = np.load(Path(DATA_DIR).joinpath('rsa', f'matrix_cka_{layer_name}_{dataset_name}_240911.npy'))
    group_dis_mean, group_dis_std = get_model_group_dist(tasks_dict, sim_matrix)
    
    dis2ref_mean_list.append(group_dis_mean[ref_group_idx])
    dis2ref_std_list.append(group_dis_std[ref_group_idx])

dis2ref_mean = np.stack(dis2ref_mean_list, axis=1) # shape (num_tasks, num_layers)
dis2ref_std = np.stack(dis2ref_std_list, axis=1)


# %%
task_name2str = {
    'Distance': 'distance_reg',
    'Translation': 'translation_reg',
    'Rotation': 'rotation_reg',
    'Dis. Tra. Rot.': 'distance_translation_rotation',
    'Obj. Category': 'category_class',
    'All spatial + cla.': 'cat_obj_class_all_latents',
}
region_list = ['V1', 'V2', 'V4', 'IT', 'Behavior']

task_region_layer_idx = {}
for task_n, task_s in task_name2str.items():
    region_layer_idx = {}
    for region in region_list:
        layer_list = df_mt[(df_mt['exp_group'] == task_s) & (df_mt['benchmark_region'] == region)]['mapped_layer'].tolist()
        region_layer_idx[region] = [layer2idx[layer] for layer in layer_list]
    task_region_layer_idx[task_n] = region_layer_idx

# add image net trained models
region_layer_idx = {}
for region in region_list:
    layer_list = df_imn[df_imn['benchmark_region'] == region]['mapped_layer'].tolist()
    region_layer_idx[region] = [layer2idx[layer] for layer in layer_list]
task_region_layer_idx['ImageNet-1K'] = region_layer_idx

# add untrained models
region_layer_idx = {}
for region in region_list:
    layer_list = df_pt_rnd[(df_pt_rnd['exp_group'] == 'random') & (df_pt_rnd['benchmark_region'] == region)]['mapped_layer'].tolist()
    region_layer_idx[region] = [layer2idx[layer] for layer in layer_list]
task_region_layer_idx['Untrained'] = region_layer_idx

# %%
# Add small offsets to repeated values for better visualization
x_offset = 0.1
offset_task_region_layer_idx = {}
for task in task_region_layer_idx:    
    region_layer_idx = task_region_layer_idx[task]
    offset_region_layer_idx = {}
    for region in region_layer_idx:
        layer_idx_list = region_layer_idx[region]
        new_list = []
        seen = {}
        for val in layer_idx_list:
            if val not in seen:
                seen[val] = 0
                new_list.append(val)
            else:
                seen[val] += 1
                new_list.append(val + seen[val] * x_offset)
        offset_region_layer_idx[region] = new_list
    offset_task_region_layer_idx[task] = offset_region_layer_idx

# %%
for task in offset_task_region_layer_idx:
    fig, ax = plt.subplots(2, 1, height_ratios=(1, 3), sharex=True, figsize=(5, 4.5))

    # plot the layer assignment
    y_offset = 0.0
    for region, layer_idx_list in offset_task_region_layer_idx[task].items():
        ax[0].scatter(layer_idx_list, np.zeros_like(layer_idx_list) + y_offset, label=region, alpha=0.5)
        y_offset -= 1.0

    for x in xaxis_layer_idx_list:
        ax[0].axvline(x=x, color='grey', linestyle=':', alpha=0.3)
    ax[0].set_title(f'Layer assignment for {task} models')
    ax[0].set_yticks([])
    ax[0].set_ylim(-4.5, 0.5)
    ax[0].legend(loc=(-0.18, 0.05), fontsize='xx-small')

    # plot the similarity to reference model
    cka_xaxis = [layer2idx[layer] for layer in record_layers]
    ax[1].errorbar(cka_xaxis, dis2ref_mean[ref_group_idx], yerr=dis2ref_std[ref_group_idx], fmt='o-', label=f'{ref_group} intra-group CKA', capsize=3, alpha=0.8, color='grey')
    if task != ref_group:
        ax[1].errorbar(cka_xaxis, dis2ref_mean[labels.index(task)], yerr=dis2ref_std[labels.index(task)], fmt='o-', label=f'{task} to {ref_group} CKA', capsize=3, alpha=0.8, color='#2A4494')

    for x in xaxis_layer_idx_list:
        ax[1].axvline(x=x, color='grey', linestyle=':', alpha=0.3)
    ax[1].set_xlabel('Layer names ( --> from early to late)')
    ax[1].set_ylabel('Similarity (CKA) to Obj. Category')
    ax[1].set_xlim(0, 22)
    ax[1].set_ylim(0.0, 1.0)
    ax[1].set_yticks([0.0, 0.5, 1.0])
    ax[1].set_xticks(xaxis_layer_idx_list, xaxis_layer_list, rotation=40, ha='right')
    ax[1].legend(loc='lower left', fontsize='small')
    fig.tight_layout()
    fig.savefig(Path(FIG_DIR).joinpath(f'layer_assign_vis_{task}.pdf'), transparent=True, bbox_inches='tight')

# %%



