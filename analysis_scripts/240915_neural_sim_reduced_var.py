# %%
from pathlib import Path
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

from dataset import TDWDataset
from utils import prepare_pytorch_model
from activity import get_model_activations
from config_global import DEVICE, EXP_DIR, DATA_DIR
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
for run_id in range(24):
    get_model_act(Path(EXP_DIR).joinpath('ctrl_var_target_dist_240712', f'run_{run_id:04d}'), dataset, record_layers, 674)
for run_id in range(20, 40):
    get_model_act(Path(EXP_DIR).joinpath('ctrl_trans_var_240814', f'run_{run_id:04d}'), dataset, record_layers, 674)

# %%
### IDEALY, SHOULD USE THIS ###
# 'multi_task_tdw_1m20240206_0718' and 'ctrl_var_target_dist_240712' have different training batches
# 'ctrl_var_target_dist_240712' and 'ctrl_trans_var_240814' have the same training batches
# thus not using 'multi_task_tdw_1m20240206_0718' models here
tasks_maps = {'Dis. reg.': ('ctrl_var_target_dist_240712', [i for i in range(3)]),
              'Tra. reg.': ('ctrl_var_target_dist_240712', [i for i in range(3, 6)]),
              'Rot reg.': ('ctrl_var_target_dist_240712', [i for i in range(6, 9)]),
              'Dis. Tra. Rot.': ('ctrl_var_target_dist_240712', [i for i in range(9, 12)]),
              'Cat. cla.': ('ctrl_trans_var_240814', [i for i in range(10, 15)]),
              'Reduced cat. var.': ('ctrl_var_target_dist_240712', [i for i in range(12, 24)]),
              'Reduced tran. var.': ('ctrl_trans_var_240814', [i for i in range(20, 40)]),
              'ImageNet cla.': ('imagenet1k_0902', [i for i in range(8)]),
              'Untrained': ('pretrain_and_random_resnet18_0220', [i for i in range(5)]),
              }

# %%
tasks_maps = {'Dis. reg.': ('multi_task_tdw_1m20240206_0718', [i for i in range(8)]),
              'Tra. reg.': ('multi_task_tdw_1m20240206_0718', [i for i in range(8, 16)]),
              'Rot reg.': ('multi_task_tdw_1m20240206_0718', [i for i in range(16, 24)]),
              'Dis. Tra. Rot.': ('multi_task_tdw_1m20240206_0718', [i for i in range(48, 56)]),
              'Cat. cla.': ('multi_task_tdw_1m20240206_0718', [i for i in range(56, 64)]),
              'All cla. all reg.': ('multi_task_tdw_1m20240206_0718', [i for i in range(72, 80)]),
              'ImageNet cla.': ('imagenet1k_0902', [i for i in range(8)]),
              'Untrained': ('pretrain_and_random_resnet18_0220', [i for i in range(5)]),
              'Reduced cat. var.': ('ctrl_var_target_dist_240712', [i for i in range(12, 24)]),
              'Reduced tran. var.': ('ctrl_trans_var_240814', [i for i in range(20, 40)]),
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

    np.save(Path(DATA_DIR).joinpath('rsa', f'matrix_cka_{layer_name}_{dataset_name}_240916.npy'), sim_matrix)

# %%
for layer in record_layers:
    layer_name = layer.replace('.', '_')

    fig, ax = plt.subplots()
    sim_matrix = np.load(Path(DATA_DIR).joinpath('rsa', f'matrix_cka_{layer_name}_{dataset_name}_240916.npy'))
    im = ax.imshow(sim_matrix, cmap='viridis')
    cbar = fig.colorbar(im, ax=ax)
    ax.set_title(f'Similarity of RDMS at {layer}')

# %%
def get_model_group_dist(tasks_dict, model_dis_mat):
    """
    calculate the mean and std of the distance between the activations of different tasks
    :param tasks_dict: dict, the dict of tasks, the key is the task name, the value is the list of indices of the task
    :param model_dis_mat: np.ndarray, the distance matrix of the model activations, must be a symmetric matrix
    """
    group_dis_mean = np.zeros((len(tasks_dict), len(tasks_dict)))
    group_dis_std = np.zeros((len(tasks_dict), len(tasks_dict)))

    for i, task1 in enumerate(tasks_dict.keys()):
        for j, task2 in enumerate(tasks_dict.keys()):
            dis_mat_sect = model_dis_mat[np.ix_(tasks_dict[task1], tasks_dict[task2])]
            if i == j:
                # calculate the intra-group similarity, use only the off-diagonal elements
                self_dis = dis_mat_sect[np.where(~np.eye(dis_mat_sect.shape[0], dtype=bool))]
                if len(self_dis) == 0:
                    group_dis_mean[i, i] = 1
                    group_dis_std[i, i] = 0
                else:
                    group_dis_mean[i, i] = np.mean(self_dis)
                    group_dis_std[i, i] = np.std(self_dis, ddof=1)
            else:
                # calculate the inter-group similarity, use all elements
                cross_dis = dis_mat_sect.flatten()
                group_dis_mean[i, j] = np.mean(cross_dis)
                if len(cross_dis) == 1:
                    group_dis_std[i, j] = 0.0
                else:
                    group_dis_std[i, j] = np.std(cross_dis, ddof=1)
    return group_dis_mean, group_dis_std

# %%
def show_group_CKA_matrix(tasks_dict, str_fix, fig_size=(3.6, 2.7)):
    labels = list(tasks_dict.keys())

    # make the heatmap of the model similarity
    for i, layer in enumerate(record_layers):
        layer_name = layer.replace('.', '_')
        sim_matrix = np.load(Path(DATA_DIR).joinpath('rsa', f'matrix_cka_{layer_name}_{dataset_name}_240916.npy'))
        group_dis_mean, group_dis_std = get_model_group_dist(tasks_dict, sim_matrix)

        fig, ax = plt.subplots(figsize=fig_size)
        sns.heatmap(group_dis_mean, cmap='Blues',
                    annot=True, fmt=".2f", annot_kws={'fontsize': 'xx-small'}, linewidths=0.5,
                    cbar_kws={'label': 'Mean similarity (CKA)'}, square=True,
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.xaxis.tick_top()
        ax.set_xticks(np.arange(len(labels)) + 0.5, labels, rotation=40, ha='left')
        ax.set_title(f'Model similarity at {layer}')
        fig.savefig(f'./figures/between_group_model_similarity_heat_{str_fix}_{layer_name}.pdf', transparent=True, bbox_inches='tight')


# %%
tasks_dict = {'Dis. reg.': [i for i in range(8)],
              'Tra. reg.': [i for i in range(8, 16)],
              'Rot reg.': [i for i in range(16, 24)],
              'Dis. Tra. Rot.': [i for i in range(24, 32)],
              'Cat. cla.': [i for i in range(32, 40)],
              'All cla. all reg.': [i for i in range(40, 48)],
              'ImageNet cla.': [i for i in range(48, 56)],
              'Untrained': [i for i in range(56, 61)],
              'Dis. reg. Reduced cat. var.': [i for i in range(61, 64)],
              'Tra. reg. Reduced cat. var.': [i for i in range(64, 67)],
              'Rot reg. Reduced cat. var.': [i for i in range(67, 70)],
              'Dis. Tra. Rot. Reduced cat. var.': [i for i in range(70, 73)],
              'Dis. reg. Reduced tran. var.': [i for i in range(73, 78)],
              'Rot. reg. Reduced tran. var.': [i for i in range(78, 83)],
              'Cat. cla. Reduced tran. var.': [i for i in range(83, 88)],
              'Obj. cla. Reduced tran. var.': [i for i in range(88, 93)],
              }
show_group_CKA_matrix(tasks_dict, 'full', (8, 6))

# %%
tasks_dict = {'Dis. reg.': [i for i in range(8)],
              'Tra. reg.': [i for i in range(8, 16)],
              'Dis. reg. Reduced cat. var.': [i for i in range(61, 64)],
              'Tra. reg. Reduced cat. var.': [i for i in range(64, 67)],
              }
show_group_CKA_matrix(tasks_dict, 'dis_tran_reduce_cat', (2.8, 2.1))

# %%
tasks_dict = {'Dis. reg.': [i for i in range(8)],
              'Rot reg.': [i for i in range(16, 24)],
              'Dis. reg. Reduced cat. var.': [i for i in range(61, 64)],
              'Rot reg. Reduced cat. var.': [i for i in range(67, 70)],
              }
show_group_CKA_matrix(tasks_dict, 'dis_rot_reduce_cat', (2.8, 2.1))

# %%
tasks_dict = {
              'Tra. reg.': [i for i in range(8, 16)],
              'Rot reg.': [i for i in range(16, 24)],
              'Tra. reg. Reduced cat. var.': [i for i in range(64, 67)],
              'Rot reg. Reduced cat. var.': [i for i in range(67, 70)],
              }
show_group_CKA_matrix(tasks_dict, 'tra_rot_reduce_cat', (2.8, 2.1))

# %%
tasks_dict = {'Dis. reg.': [i for i in range(8)],
              'Rot reg.': [i for i in range(16, 24)],
              'Dis. reg. Reduced tran. var.': [i for i in range(73, 78)],
              'Rot. reg. Reduced tran. var.': [i for i in range(78, 83)],
              }
show_group_CKA_matrix(tasks_dict, 'dis_rot_reduce_tra', (2.8, 2.1))

# %%
tasks_dict = {'Dis. reg.': [i for i in range(8)],
              'Cat. cla.': [i for i in range(32, 40)],
              'Dis. reg. Reduced tran. var.': [i for i in range(73, 78)],
              'Cat. cla. Reduced tran. var.': [i for i in range(83, 88)],
              }
show_group_CKA_matrix(tasks_dict, 'dis_cat_reduce_tra', (2.8, 2.1))

# %%
tasks_dict = {
              'Rot reg.': [i for i in range(16, 24)],
              'Cat. cla.': [i for i in range(32, 40)],
              'Rot. reg. Reduced tran. var.': [i for i in range(78, 83)],
              'Cat. cla. Reduced tran. var.': [i for i in range(83, 88)],
              }
show_group_CKA_matrix(tasks_dict, 'rot_cat_reduce_tra', (2.8, 2.1))

# %%



