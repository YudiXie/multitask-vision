# %%
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

import torchvision.transforms as transforms
import torch
from torchvision.models import resnet18
from sklearn.manifold import MDS
from tqdm import trange

import matplotlib.pyplot as plt
import seaborn as sns
import easyfigs.heatmap as hm

import easyfigs.basicplot as bp
from dataset import TDWDataset
from utils import prepare_pytorch_model
from activity import get_model_activations
from config_global import DEVICE, EXP_DIR, DATA_DIR, FIG_DIR
from cka import linear_CKA

# %%
# model must be a resnet
def remove_resnet_duplicates(activity_dict):
    # reduce the duplicate activations in resnet
    # because the later relu layer are used twice in resnet,
    for k, v in activity_dict.items():
        if '.relu' in k:
            v.pop(-2)


def get_model_act(run_path, dataset, record_layers, out_num):
    """
    record the activations of the model on the dataset on the specified layers
    :param run_path: the path to save the activations
    :param dataset: pytorch dataset object, the dataset to record the activations
    :param record_layers: the layers to record
    """
    npy_path_dict = {}
    for layer in record_layers:
        layer_name = layer.replace('.', '_')
        npy_path_dict[layer] = run_path.joinpath(f'act_{layer_name}_{dataset.dset_name}.npy')

    has_records = [True if npy_path.is_file() else False for npy_path in npy_path_dict.values()]
    
    if all(has_records):
        print('All activations are already recorded!')
        all_activations = {}
        for layer, npy_path in npy_path_dict.items():
            all_activations[layer] = np.load(npy_path)
        return all_activations
    else:
        print('Start recording activations ...')
        model = prepare_pytorch_model('resnet18', out_num, run_path.joinpath('model.pth'))
        all_activations = get_model_activations(dataset, model, record_layers, remove_resnet_duplicates)
        for layer, npy_path in npy_path_dict.items():
            if npy_path.is_file():
                npy_path.unlink()
            np.save(npy_path, all_activations[layer])
        return all_activations

# %%
# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

dataset = TDWDataset(root_dir='/om/user/yu_xie/data/tdw_images/tdw_1m_20240206',
                     split='val', fraction=0.04, transform=transform)

record_layers = ['layer1.0.relu', 'layer2.0.relu', 'layer3.0.relu', 'layer4.0.relu']

# %%
# extract and save model activations
for run_id in range(48):
    get_model_act(Path(EXP_DIR).joinpath('ctrl_cat_var_240927', f'run_{run_id:04d}'), dataset, record_layers, 674)
for run_id in range(48):
    get_model_act(Path(EXP_DIR).joinpath('ctrl_trans_var_240927', f'run_{run_id:04d}'), dataset, record_layers, 674)

# %%
tasks_maps = {'Dis. reg.': ('ctrl_cat_var_240927', [i for i in range(6)]), # 0,6
              'Tra. reg.': ('ctrl_cat_var_240927', [i for i in range(6, 12)]), # 6,12
              'Rot reg.': ('ctrl_cat_var_240927', [i for i in range(12, 18)]), # 12,18
              'Dis. Rot.': ('ctrl_trans_var_240927', [i for i in range(12, 18)]), # 18,24
              'Dis. Tra. Rot.': ('ctrl_cat_var_240927', [i for i in range(18, 24)]), # 24, 30
              'Cat. cla.': ('ctrl_trans_var_240927', [i for i in range(18, 24)]), # 30, 36
              'Reduced cat. var.': ('ctrl_cat_var_240927', [i for i in range(24, 48)]), # 36, 60
              'Reduced tran. var.': ('ctrl_trans_var_240927', [i for i in range(24, 48)]), # 60, 84
              }

# %%
task_name_list = []
exp_name_list = []
run_id_save_list = []
for task_name, (exp_name, run_id_list) in tasks_maps.items():
    for run_id in run_id_list:
        task_name_list.append(task_name)
        exp_name_list.append(exp_name)
        run_id_save_list.append(run_id)
model_df = pd.DataFrame.from_dict({'task': task_name_list, 'exp_name': exp_name_list, 'run_id': run_id_save_list})

# %%
matrix_suffix = '240928'

dataset_name = 'tdw_1m_20240206_val_0_04'
for layer in record_layers:
    print(f'Processing layer {layer} ...')
    layer_name = layer.replace('.', '_')

    # compute the model-model similarity matrix
    sim_matrix = np.zeros((len(model_df), len(model_df)))
    for i in trange(len(model_df)):
        for j in range(i, len(model_df)):
            exp_name1, run_id1 = model_df.iloc[i]['exp_name'], model_df.iloc[i]['run_id']
            exp_name2, run_id2 = model_df.iloc[j]['exp_name'], model_df.iloc[j]['run_id']
            model_id1, model_id2 = f'{exp_name1}_{run_id1:04d}', f'{exp_name2}_{run_id2:04d}'
            
            score_path = Path(DATA_DIR).joinpath('rsa', f'cka_{model_id1}_{model_id2}_{layer_name}_{dataset_name}.npy')
            if score_path.is_file():
                CKA_score = np.load(score_path)
            else:
                act1 = np.load(Path(EXP_DIR).joinpath(exp_name1, f'run_{run_id1:04d}', f'act_{layer_name}_{dataset_name}.npy'))
                act2 = np.load(Path(EXP_DIR).joinpath(exp_name2, f'run_{run_id2:04d}', f'act_{layer_name}_{dataset_name}.npy'))
                CKA_score = linear_CKA(act1, act2)
                np.save(score_path, CKA_score)

            sim_matrix[i, j] = CKA_score
            sim_matrix[j, i] = sim_matrix[i, j]

    np.save(Path(DATA_DIR).joinpath('rsa', f'matrix_cka_{layer_name}_{dataset_name}_{matrix_suffix}.npy'), sim_matrix)

# %%
def plot_model_spread_layer(sim_matrix, exp_name, layer, name_group_full, task_group_full, name_group_redu, task_group_redu):
    task_list = list(task_group_full.keys())

    full_var_mean = []
    full_var_std = []
    redu_var_mean = []
    redu_var_std = []

    for task in task_group_full.keys():
        group_full_idx = task_group_full[task]
        sim_group_full = sim_matrix[np.ix_(group_full_idx, group_full_idx)]
        off_d_sim_group_full = sim_group_full[np.where(~np.eye(sim_group_full.shape[0], dtype=bool))]
        full_var_mean.append(np.mean(off_d_sim_group_full))
        full_var_std.append(np.std(off_d_sim_group_full, ddof=1))

        group_redu_idx = task_group_redu[task]
        sim_group_redu = sim_matrix[np.ix_(group_redu_idx, group_redu_idx)]
        off_d_sim_group_redu = sim_group_redu[np.where(~np.eye(sim_group_redu.shape[0], dtype=bool))]
        redu_var_mean.append(np.mean(off_d_sim_group_redu))
        redu_var_std.append(np.std(off_d_sim_group_redu, ddof=1))

    plot_data = {
        name_group_full: {'y': full_var_mean, 'error': full_var_std, 'kwargs': {'color': 'C1', 'alpha': 0.8}},
        name_group_redu: {'y': redu_var_mean, 'error': redu_var_std, 'kwargs': {'color': 'grey'}},
    }

    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    x_axis, _ignore = bp.bar_groups(ax, task_list, plot_data, bar_label=True)
    x_s, x_e = ax.get_xlim()
    ax.set_xticks(x_axis, task_list, rotation=-20)
    ax.set_ylabel('Intra-group similarity (CKA)')
    ax.set_xlim(x_s, x_e)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc='lower left', fontsize='small')
    ax.set_title(f'{layer}', y=1.07)
    bp.remove_top_right_spines(ax)
    fig.tight_layout()
    fig.savefig(Path(FIG_DIR).joinpath(f'model_spread_{exp_name}.pdf'), transparent=True, bbox_inches='tight')

# %%
# task group with full variability
task_group_full = {
    'Distance': [i for i in range(6)],
    'Translation': [i for i in range(6, 12)],
    'Rotation': [i for i in range(12, 18)],
    'Dis. Tra. Rot.': [i for i in range(24, 30)],
}
# task group with reducded category variability
task_group_redu = {
    'Distance': [i for i in range(36, 42)],
    'Translation': [i for i in range(42, 48)],
    'Rotation': [i for i in range(48, 54)],
    'Dis. Tra. Rot.': [i for i in range(54, 60)],
}

for layer in record_layers:
    layer_name = layer.replace('.', '_')
    sim_matrix = np.load(Path(DATA_DIR).joinpath('rsa', f'matrix_cka_{layer_name}_{dataset_name}_{matrix_suffix}.npy'))

    plot_model_spread_layer(sim_matrix, f'cat_var_layer_{layer_name}', layer,
                            'Full cat. var.', task_group_full,
                            'Reduced cat. var.', task_group_redu)

# %%
# task group with full variability
task_group_full = {
    'Distance': [i for i in range(6)],
    'Rotation': [i for i in range(12, 18)],
    'Dis. Rot.': [i for i in range(18, 24)],
    'Category': [i for i in range(30, 36)],
}
# task group with reducded translation variability
task_group_redu = {
    'Distance': [i for i in range(60, 66)],
    'Rotation': [i for i in range(66, 72)],
    'Dis. Rot.': [i for i in range(72, 78)],
    'Category': [i for i in range(78, 84)],
}

for layer in record_layers:
    layer_name = layer.replace('.', '_')
    sim_matrix = np.load(Path(DATA_DIR).joinpath('rsa', f'matrix_cka_{layer_name}_{dataset_name}_{matrix_suffix}.npy'))

    plot_model_spread_layer(sim_matrix, f'tran_var_layer_{layer_name}', layer,
                            'Full tran. var.', task_group_full,
                            'Reduced tran. var.', task_group_redu)

# %%
def plot_dist_to_ref_reduced_var(ref_group, ref_idx, task_group_full, task_group_redu, clist):
    f_dis2ref_mean = defaultdict(list)
    f_dis2ref_std = defaultdict(list)
    r_dis2ref_mean = defaultdict(list)
    r_dis2ref_std = defaultdict(list)
    inter_ref_sim_mean = []
    inter_ref_sim_std = []

    # visualize similarity to reference model
    for i, layer in enumerate(record_layers):
        layer_name = layer.replace('.', '_')
        sim_matrix = np.load(Path(DATA_DIR).joinpath('rsa', f'matrix_cka_{layer_name}_{dataset_name}_{matrix_suffix}.npy'))

        sim_ref = sim_matrix[np.ix_(ref_idx, ref_idx)]
        off_d_sim_ref = sim_ref[np.where(~np.eye(sim_ref.shape[0], dtype=bool))]
        inter_ref_sim_mean.append(np.mean(off_d_sim_ref))
        inter_ref_sim_std.append(np.std(off_d_sim_ref, ddof=1))

        for task in task_group_full.keys():
            group_sim_f = sim_matrix[np.ix_(task_group_full[task], ref_idx)]
            group_sim_r = sim_matrix[np.ix_(task_group_redu[task], ref_idx)]

            f_dis2ref_mean[task].append(np.mean(group_sim_f))
            f_dis2ref_std[task].append(np.std(group_sim_f, ddof=1))
            r_dis2ref_mean[task].append(np.mean(group_sim_r))
            r_dis2ref_std[task].append(np.std(group_sim_r, ddof=1))

    task_list = list(task_group_full.keys())
    x_axis = np.arange(len(record_layers))
    fig, ax = plt.subplots(figsize=(6, 2.8))
    offset = 0.08 * 3
    full_redu_offset = 0.08
    offset_step = 0
    for i, task in enumerate(task_list):
        task_x = x_axis + offset_step * offset
        ax.errorbar(task_x, f_dis2ref_mean[task], yerr=f_dis2ref_std[task], 
                    fmt='o', capsize=3, label=task, color=clist[i])
        ax.errorbar(task_x + full_redu_offset, r_dis2ref_mean[task], yerr=r_dis2ref_std[task], 
                    fmt='P', capsize=3, label=task + ' reduced var.', color=clist[i])
        offset_step += 1

    # add the intra-group similarity as the reference 
    x_end = x_axis + (len(task_list) - 1) * offset + full_redu_offset
    ax.hlines(inter_ref_sim_mean, x_axis - full_redu_offset, x_end + full_redu_offset,
            linestyles='dashed', label=f'intra - {ref_group}', color='k')
    for i in range(len(record_layers)):
        ax.fill_between([x_axis[i] - full_redu_offset, x_end[i] + full_redu_offset], 
                        2 * [inter_ref_sim_mean[i] - inter_ref_sim_std[i]], 
                        2 * [inter_ref_sim_mean[i] + inter_ref_sim_std[i]], 
                        alpha=0.2, color='k')

    # add the vertical lines to separate the layers
    tick_centers = x_axis + ((len(task_list) - 1) * offset + full_redu_offset) / 2
    ax.set_xticks(tick_centers, record_layers)
    for i in range(len(tick_centers) - 1):
        ax.vlines(tick_centers[i] + 0.5, 0, 1, linestyles=':', color='k', alpha=0.1)

    ax.set_ylabel(f'Similarity to {ref_group} (CKA)')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xlim(tick_centers[0] - 0.5, tick_centers[-1] + 0.5)
    ax.legend(fontsize='small', loc=(1.02, 0.15))
    ref_group_save = ref_group.replace(' ', '_')
    fig.savefig(f'./figures/reduced_var_dist_to_ref_{ref_group_save}.pdf', transparent=True, bbox_inches='tight')

# %%
# visualize similarity to reference model
ref_group = 'Category'
ref_idx = [i for i in range(30, 36)]

color_list = ['#448aff', '#1565c0', '#009688', '#8bc34a']

# task group with full variability
task_group_full = {
    'Distance': [i for i in range(6)],
    'Translation': [i for i in range(6, 12)],
    'Rotation': [i for i in range(12, 18)],
    'Dis. Tra. Rot.': [i for i in range(24, 30)],
}
# task group with reducded category variability
task_group_redu = {
    'Distance': [i for i in range(36, 42)],
    'Translation': [i for i in range(42, 48)],
    'Rotation': [i for i in range(48, 54)],
    'Dis. Tra. Rot.': [i for i in range(54, 60)],
}

plot_dist_to_ref_reduced_var(ref_group, ref_idx, task_group_full, task_group_redu, color_list)

# %%
# visualize similarity to reference model
ref_group = 'Translation'
ref_idx = [i for i in range(6, 12)]

color_list = ['#448aff', '#009688', '#D05BEB', '#ffc107']

# task group with full variability
task_group_full = {
    'Distance': [i for i in range(6)],
    'Rotation': [i for i in range(12, 18)],
    'Dis. Rot.': [i for i in range(18, 24)],
    'Category': [i for i in range(30, 36)],
}
# task group with reducded translation variability
task_group_redu = {
    'Distance': [i for i in range(60, 66)],
    'Rotation': [i for i in range(66, 72)],
    'Dis. Rot.': [i for i in range(72, 78)],
    'Category': [i for i in range(78, 84)],
}

plot_dist_to_ref_reduced_var(ref_group, ref_idx, task_group_full, task_group_redu, color_list)

# %%



