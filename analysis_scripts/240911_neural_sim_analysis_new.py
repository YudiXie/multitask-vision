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
for run_id in range(80):
    get_model_act(Path(EXP_DIR).joinpath('multi_task_tdw_1m20240206_0718', f'run_{run_id:04d}'), dataset, record_layers, 674)
for run_id in range(6):
    get_model_act(Path(EXP_DIR).joinpath('pretrain_and_random_resnet18_0220', f'run_{run_id:04d}'), dataset, record_layers, 674)
for run_id in range(8):
    get_model_act(Path(EXP_DIR).joinpath('imagenet1k_0902', f'run_{run_id:04d}'), dataset, record_layers, 1000)

# %%
# get simclr model activations
# simclr models from https://github.com/sthalles/SimCLR
def get_simclr_model_act(run_path, dataset, record_layers):
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
        loaded_dict = torch.load(run_path.joinpath('checkpoint_0100.pth.tar'), map_location=DEVICE)
        state_dict = {}
        for k, v in loaded_dict['state_dict'].items():
            state_dict[k[9:]] = v
        
        model = resnet18()
        model = model.to(DEVICE)
        mismatch = model.load_state_dict(state_dict, strict=False)
        print(mismatch)

        all_activations = get_model_activations(dataset, model, record_layers, remove_resnet_duplicates)
        for layer, npy_path in npy_path_dict.items():
            if npy_path.is_file():
                npy_path.unlink()
            np.save(npy_path, all_activations[layer])
        return all_activations

# %%
get_simclr_model_act(Path(EXP_DIR).joinpath('simclr_models_0911', 'run_0000'), dataset, record_layers)
get_simclr_model_act(Path(EXP_DIR).joinpath('simclr_models_0911', 'run_0001'), dataset, record_layers)

# %%
tasks_maps = {'Dis. reg.': ('multi_task_tdw_1m20240206_0718', [i for i in range(8)]),
              'Tra. reg.': ('multi_task_tdw_1m20240206_0718', [i for i in range(8, 16)]),
              'Rot reg.': ('multi_task_tdw_1m20240206_0718', [i for i in range(16, 24)]),
              'Dis. Tra. Rot.': ('multi_task_tdw_1m20240206_0718', [i for i in range(48, 56)]),
              'Cat. cla.': ('multi_task_tdw_1m20240206_0718', [i for i in range(56, 64)]),
              'All cla. all reg.': ('multi_task_tdw_1m20240206_0718', [i for i in range(72, 80)]),
              'ImageNet cla.': ('imagenet1k_0902', [i for i in range(8)]),
            #   'SimCLR STL10': ('simclr_models_0911', [0]),
            #   'SimCLR CIFAR10': ('simclr_models_0911', [1]),
              'Untrained': ('pretrain_and_random_resnet18_0220', [i for i in range(5)]),
              }

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

    np.save(Path(DATA_DIR).joinpath('rsa', f'matrix_cka_{layer_name}_{dataset_name}_240911.npy'), sim_matrix)

# %%
for layer in record_layers:
    layer_name = layer.replace('.', '_')

    fig, ax = plt.subplots()
    sim_matrix = np.load(Path(DATA_DIR).joinpath('rsa', f'matrix_cka_{layer_name}_{dataset_name}_240911.npy'))
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
tasks_dict = {'Dis. reg.': [i for i in range(8)],
              'Tra. reg.': [i for i in range(8, 16)],
              'Rot reg.': [i for i in range(16, 24)],
              'Dis. Tra. Rot.': [i for i in range(24, 32)],
              'Cat. cla.': [i for i in range(32, 40)],
              'All cla. all reg.': [i for i in range(40, 48)],
              'ImageNet cla.': [i for i in range(48, 56)],
              'Untrained': [i for i in range(56, 61)],
              }
labels = list(tasks_dict.keys())

# %%
# make the heatmap of the model similarity
for i, layer in enumerate(record_layers):
    layer_name = layer.replace('.', '_')
    sim_matrix = np.load(Path(DATA_DIR).joinpath('rsa', f'matrix_cka_{layer_name}_{dataset_name}_240911.npy'))
    group_dis_mean, group_dis_std = get_model_group_dist(tasks_dict, sim_matrix)
    group_dis_mean = np.tril(group_dis_mean)

    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    sns.heatmap(group_dis_mean, vmin=0.0, vmax=1.0, cmap='Blues',
                annot=True, fmt=".2f", annot_kws={'fontsize': 'xx-small'}, linewidths=0.5,
                cbar_kws={'label': 'Mean similarity (CKA)'}, square=True,
                xticklabels=labels, yticklabels=labels, ax=ax)
    # ax.xaxis.tick_top()
    ax.set_xticks(np.arange(len(labels)) + 0.5, labels, rotation=40, ha='right')
    ax.set_title(f'Model similarity at {layer}')
    fig.savefig(f'./figures/between_group_model_similarity_heat_{layer_name}.pdf', transparent=True, bbox_inches='tight')

# %%
# make figures that show individual rows in the matrix
for i, layer in enumerate(record_layers):
    layer_name = layer.replace('.', '_')
    sim_matrix = np.load(Path(DATA_DIR).joinpath('rsa', f'matrix_cka_{layer_name}_{dataset_name}_240911.npy'))
    group_dis_mean, group_dis_std = get_model_group_dist(tasks_dict, sim_matrix)
    
    # make figures that show individual rows in the matrix
    x_axis = np.arange(len(labels))
    fig, axes = plt.subplots(8, 1, figsize=(3, 15), sharex=True, sharey=True)
    for i, task in enumerate(labels):
        ax = axes[i]
        ele1 = ax.errorbar(x_axis, group_dis_mean[i], yerr=group_dis_std[i], fmt='o', capsize=3, label='Between-group sim.')

        intra_mean = group_dis_mean[i, i]
        intra_std = group_dis_std[i, i]
        ele2 = ax.hlines(intra_mean, x_axis[0], x_axis[-1], linestyles='dashed', label='Intra-group sim.', color='k')
        ax.fill_between([x_axis[0], x_axis[-1]], 2 * [intra_mean - intra_std], 2 * [intra_mean + intra_std], alpha=0.2, color='k')
        
        ax.set_title(f'{task} vs. Others', fontsize=10)
        if i == 7:
            ax.set_xticks(x_axis, labels, rotation=90)
    
    # fig.legend(handles=[ele1, ele2], loc=(0.58, 0.02), ncols=2)
    fig.suptitle(f'Model similarity at {layer}')
    fig.supxlabel('Training tasks')
    fig.supylabel('Representational similarity (CKA)')
    fig.tight_layout()
    fig.savefig(f'./figures/between_group_model_similarity_rows_{layer_name}.pdf', transparent=True, bbox_inches='tight')

# %%
markers = ['o', 's', 'P', 'X', '*', 'p', 'd', '^']
color_list = ['448aff', '1565c0', '009688', '8bc34a', 'ffc107', 'ff9800', 'f44336', 'ad1457']

# %%
X_transformed = {}
for i, layer in enumerate(record_layers):
    layer_name = layer.replace('.', '_')
    sim_matrix = np.load(Path(DATA_DIR).joinpath('rsa', f'matrix_cka_{layer_name}_{dataset_name}_240911.npy'))

    embedding = MDS(n_components=2, normalized_stress='auto')
    X_transformed[layer] = embedding.fit_transform(sim_matrix)



fig, axes = plt.subplots(1, 4, figsize=(10, 3.2))
for i, layer in enumerate(record_layers):
    ax = axes[i]
    plot_data = X_transformed[layer]
    eles = []
    for j, (task, indices) in enumerate(tasks_dict.items()):
        ele1 = ax.scatter(plot_data[indices, 0], plot_data[indices, 1],
                          label=task, alpha=0.8, marker=markers[j], s=50, c=f'#{color_list[j]}')
        eles.append(ele1)
    ax.set_title(f'{layer}')
    ax.set_xticks([])
    ax.set_yticks([])
fig.legend(handles=eles, fontsize='x-small', ncols=4, loc=(0.57, 0.03))
# fig.legend(handles=[ele1, ele2], loc=(0.58, 0.02), ncols=2)
fig.suptitle('MDS of model similarity matrix')
fig.supxlabel('MDS dim. 1')
fig.supylabel('MDS dim. 2')
fig.tight_layout()
fig.savefig('./figures/mds_model_cka.pdf', transparent=True, bbox_inches='tight')


# %%
# visualize similarity to reference model
def plot_dist_to_ref_group(ref_group):
    ref_group_idx = labels.index(ref_group)

    dis2ref_mean_list = []
    dis2ref_std_list = []
    for i, layer in enumerate(record_layers):
        layer_name = layer.replace('.', '_')
        sim_matrix = np.load(Path(DATA_DIR).joinpath('rsa', f'matrix_cka_{layer_name}_{dataset_name}_240911.npy'))
        group_dis_mean, group_dis_std = get_model_group_dist(tasks_dict, sim_matrix)
        
        dis2ref_mean_list.append(group_dis_mean[ref_group_idx])
        dis2ref_std_list.append(group_dis_std[ref_group_idx])

    dis2ref_mean = np.stack(dis2ref_mean_list, axis=1)
    dis2ref_std = np.stack(dis2ref_std_list, axis=1)


    x_axis = np.arange(len(record_layers))
    fig, ax = plt.subplots(figsize=(6, 2.8))
    offset = 0.08
    offset_step = 0
    for i in range(len(labels)):
        if i == ref_group_idx:
            pass
        else:
            ax.errorbar(x_axis + offset_step * offset, dis2ref_mean[i], yerr=dis2ref_std[i], 
                        fmt='o', capsize=3, label=labels[i], color=f'#{color_list[i]}')
            offset_step += 1

    # add the intra-group similarity as the reference 
    x_end = x_axis + (len(labels) - 2) * offset
    ax.hlines(dis2ref_mean[ref_group_idx], x_axis - offset, x_end + offset, 
            linestyles='dashed', label=f'intra - {ref_group}', color='k')
    for i in range(len(record_layers)):
        ax.fill_between([x_axis[i] - offset, x_end[i] + offset], 
                        2 * [dis2ref_mean[ref_group_idx][i] - dis2ref_std[ref_group_idx][i]], 
                        2 * [dis2ref_mean[ref_group_idx][i] + dis2ref_std[ref_group_idx][i]], 
                        alpha=0.2, color='k')

    # add the vertical lines to separate the layers
    tick_centers = x_axis + (len(labels) - 2) * offset / 2
    ax.set_xticks(tick_centers, record_layers)
    for i in range(len(tick_centers) - 1):
        ax.vlines(tick_centers[i] + 0.5, 0, 1, linestyles=':', color='k', alpha=0.1)

    ax.set_ylabel(f'Similarity (CKA) to {ref_group}')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xlim(tick_centers[0] - 0.5, tick_centers[-1] + 0.5)
    ax.legend(fontsize='small', loc=(1.02, 0.15))
    ref_group_save = ref_group.replace(' ', '_')
    fig.savefig(f'./figures/dist_to_{ref_group_save}.pdf', transparent=True, bbox_inches='tight')

# %%
plot_dist_to_ref_group('Cat. cla.')

# %%
plot_dist_to_ref_group('ImageNet cla.')

# %%



