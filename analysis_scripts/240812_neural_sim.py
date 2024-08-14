# %%
from pathlib import Path
import numpy as np
import rsatoolbox
import torchvision.transforms as transforms
from sklearn.manifold import MDS
from tqdm import trange

from dataset import TDWDataset
from utils import prepare_pytorch_model
from activity import get_model_activations

import easyfigs.heatmap as hm
import matplotlib.pyplot as plt

from config_global import EXP_DIR

# %%
# model must be a resnet
def remove_resnet_duplicates(activity_dict):
    # reduce the duplicate activations in resnet
    # because the later relu layer are used twice in resnet,
    for k, v in activity_dict.items():
        if '.relu' in k:
            v.pop(-2)


def get_model_act(run_path, dataset, record_layers):
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
        model = prepare_pytorch_model('resnet18', 674, run_path.joinpath('model.pth'))
        all_activations = get_model_activations(dataset, model, record_layers, remove_resnet_duplicates)
        for layer, npy_path in npy_path_dict.items():
            if npy_path.is_file():
                npy_path.unlink()
            np.save(npy_path, all_activations[layer])
        return all_activations

# %%
# remove previous activations
exp_dir = Path(EXP_DIR).joinpath('pretrain_and_random_resnet18_0220')
for i in range(10):
    run_path = exp_dir.joinpath(f'run_{i:04d}')
    for layer in ['layer1.0.relu', 'layer2.0.relu', 'layer3.0.relu', 'layer4.0.relu']:
        layer_n = layer.replace('.', '_')
        run_path.joinpath(f'act_tdw_1m_20240206_val_{layer_n}.npy').unlink()

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
    get_model_act(Path(EXP_DIR).joinpath('multi_task_tdw_1m20240206_0718', f'run_{run_id:04d}'), dataset, record_layers)
for run_id in range(6):
    get_model_act(Path(EXP_DIR).joinpath('pretrain_and_random_resnet18_0220', f'run_{run_id:04d}'), dataset, record_layers)


# %%
def get_group_rdms(exp_name, run_id_list, dataset, layer_list):
    rdm_dict = {}
    for layer in layer_list:
        rdm_dict[layer] = []
    
    for run_id in run_id_list:
        model_act = get_model_act(Path(EXP_DIR).joinpath(exp_name, f'run_{run_id:04d}'),
                                  dataset, record_layers)
        for layer in layer_list:
            data = rsatoolbox.data.Dataset(model_act[layer])
            rdm_dict[layer].append(rsatoolbox.rdm.calc_rdm(data))
    
    for layer in layer_list:
        rdm_dict[layer] = rsatoolbox.rdm.concat(rdm_dict[layer])
    return rdm_dict

# %%
rdms = {}
rdms['distance_reg'] = get_group_rdms('multi_task_tdw_1m20240206_0718', [i for i in range(8)], dataset, record_layers)
rdms['translation_reg'] = get_group_rdms('multi_task_tdw_1m20240206_0718', [i for i in range(8, 16)], dataset, record_layers)
rdms['rotation_reg'] = get_group_rdms('multi_task_tdw_1m20240206_0718', [i for i in range(16, 24)], dataset, record_layers)
rdms['distance_translation_rotation'] = get_group_rdms('multi_task_tdw_1m20240206_0718', [i for i in range(48, 56)], dataset, record_layers)
rdms['category_class'] = get_group_rdms('multi_task_tdw_1m20240206_0718', [i for i in range(56, 64)], dataset, record_layers)
rdms['cat_obj_class_all_latents'] = get_group_rdms('multi_task_tdw_1m20240206_0718', [i for i in range(72, 80)], dataset, record_layers)

rdms['random'] = get_group_rdms('pretrain_and_random_resnet18_0220', [i for i in range(5)], dataset, record_layers)
rdms['imagenet'] = get_group_rdms('pretrain_and_random_resnet18_0220', [5, ], dataset, record_layers)

# %%
# need a node with 128GB mem, 80 GB mem will crash the kernel
all_rdms = {}
for layer in record_layers:
    all_rdms[layer] = rsatoolbox.rdm.concat([rdms[key][layer] for key in rdms.keys()])

# %%
rdm_compare = {}
for layer in record_layers:
    rdm_compare[layer] = rsatoolbox.rdm.compare_cosine_cov_weighted(all_rdms[layer], all_rdms[layer])

# %%
for layer in record_layers:
    fig, ax = plt.subplots()
    im = ax.imshow(rdm_compare[layer], cmap='viridis')
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
                group_dis_std[i, j] = np.std(cross_dis, ddof=1)
    return group_dis_mean, group_dis_std

# %%
tasks_dict = {'Dis. reg.': [i for i in range(8)],
              'Tra. reg.': [i for i in range(8, 16)],
              'Rot reg.': [i for i in range(16, 24)],
              'Dis. Tra. Rot.': [i for i in range(24, 32)],
              'Cat. cla.': [i for i in range(32, 40)],
              'All cla. all reg.': [i for i in range(40, 48)],
              'ImageNet': [53, ],
              'Random': [i for i in range(48, 53)],
              }
labels = list(tasks_dict.keys())

# %%
for i, layer in enumerate(record_layers):
    group_dis_mean, group_dis_std = get_model_group_dist(tasks_dict, rdm_compare[layer])

    layer_name = layer.replace('.', '_')
    fig, ax = plt.subplots()
    im, cbar = hm.heatmap(group_dis_mean, labels, labels, ax=ax,
                          cmap="Blues", vmin=0.0, vmax=1.0,
                          cbarlabel="Representational Similarity")
    texts = hm.annotate_heatmap(im)
    ax.set_title(f'Mean between groups similarity at {layer}')
    fig.tight_layout()
    fig.savefig(f'./figures/between_group_model_similarity_heat_{layer_name}.pdf', transparent=True)

    # make figures that show individual rows in the matrix
    x_axis = np.arange(len(labels))
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharex=True, sharey=True)
    for i, task in enumerate(labels):
        ax = axes[i // 4, i % 4]
        ele1 = ax.errorbar(x_axis, group_dis_mean[i], yerr=group_dis_std[i], fmt='o', capsize=3, label='Between-group sim.')

        intra_mean = group_dis_mean[i, i]
        intra_std = group_dis_std[i, i]
        ele2 = ax.hlines(intra_mean, x_axis[0], x_axis[-1], linestyles='dashed', label='Intra-group sim.', color='k')
        ax.fill_between([x_axis[0], x_axis[-1]], 2 * [intra_mean - intra_std], 2 * [intra_mean + intra_std], alpha=0.2, color='k')
        
        ax.set_title(f'{task} vs. Others', fontsize=10)
        if i // 4 == 1:
            ax.set_xticks(x_axis, labels, rotation=90)
    
    fig.legend(handles=[ele1, ele2], loc='upper right', ncols=2)
    fig.suptitle(f'Between group model similarity at {layer}')
    fig.supxlabel('Training tasks')
    fig.supylabel('Representational Similarity')
    fig.tight_layout()
    fig.savefig(f'./figures/between_group_model_similarity_rows_{layer_name}.pdf', transparent=True)


# %%
# 216 x 216 RDMs comparision, takes less than 3 mins
layer_cat_rdms = rsatoolbox.rdm.concat([all_rdms[layer] for layer in record_layers])

# %%
def plot_mds(X_transformed, slice_size, fig_suf):
    markers = ['o', 's', 'P', 'X', '*', 'p', 'd', '^']
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharex=True, sharey=True)
    for i, layer in enumerate(record_layers):
        ax = axes[i]
        X_layer = X_transformed[i * slice_size: (i + 1) * slice_size]
        for j, (task, indices) in enumerate(tasks_dict.items()):
            ax.scatter(X_layer[indices, 0], X_layer[indices, 1],
                       label=task, alpha=0.8, marker=markers[j], s=80)
        ax.set_title(f'{layer}')
    axes[0].legend()
    fig.suptitle('MDS of model similarity')
    fig.supxlabel('MDS dim. 1')
    fig.supylabel('MDS dim. 2')
    fig.tight_layout()
    fig.savefig(f'./figures/mds_model_rdm_compare_{fig_suf}.pdf', transparent=True)

# %%
# RDM comparator: euclidean distance
all_m = layer_cat_rdms.get_matrices()
dis_mat = []
for i in trange(all_m.shape[0]):
    dis_mat.append(np.linalg.norm(all_m - all_m[i], ord='fro', axis=(1, 2)))
dis_mat = np.array(dis_mat)

# %%
l_m_size = dis_mat.shape[0] // len(record_layers)
embedding = MDS(n_components=2, normalized_stress='auto', dissimilarity='precomputed')
X_transformed = embedding.fit_transform(dis_mat)

plot_mds(X_transformed, l_m_size, 'euclidean')

# %%
# use CKA score directly as features to compute MDS
sim_mat = rsatoolbox.rdm.compare_cosine_cov_weighted(layer_cat_rdms, layer_cat_rdms)

# %%
l_m_size = dis_mat.shape[0] // len(record_layers)
embedding = MDS(n_components=2, normalized_stress='auto')
X_transformed = embedding.fit_transform(sim_mat)

plot_mds(X_transformed, l_m_size, 'cosine_cov_weighted_sim_feature')


