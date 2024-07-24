# %%
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
from activity import get_model_activations_on_dataset, cross_validate_on_target
from utils import prepare_pytorch_model
from config_global import EXP_DIR, ROOT_DIR
from pathlib import Path
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from dataset import HVMDataset, TDWDataset
from torchvision.models import resnet18, ResNet18_Weights, ResNet

from activity import get_model_activations
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

# %%
# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# model must be a resnet
def remove_resnet_duplicates(activity_dict):
    # reduce the duplicate activations in resnet
    # because the later relu layer are used twice in resnet,
    for k, v in activity_dict.items():
        if '.relu' in k:
            v.pop(-2)

dataset = TDWDataset(root_dir='/om/user/yu_xie/data/tdw_images/tdw_image_dataset_1m_20240206',
                     split='train', fraction=1e-3, transform=transform)

record_layers = ['layer1.0.relu', 'layer2.0.relu', 'layer3.0.relu', 'layer4.0.relu']

# %%
def record_model_act(run_path, record_layers):
    """
    record the activations of the model on the dataset on the specified layers
    :param run_id: the run id of the model
    :param record_layers: the layers to record
    :param overwrite: whether to overwrite the existing activations
    """
    model = prepare_pytorch_model('resnet18', 674, run_path.joinpath('model.pth'))
    all_activations = get_model_activations(dataset, model, record_layers, remove_resnet_duplicates)
    for layer in record_layers:
        layer_name = layer.replace('.', '_')
        npy_path = run_path.joinpath(f'act_tdw_1m_20240206_val_{layer_name}.npy')
        if npy_path.is_file():
            npy_path.unlink()
        np.save(npy_path, all_activations[layer])
    return all_activations


# %%
full_data_act = record_model_act(Path(EXP_DIR).joinpath('ctrl_var_target_dist_240712', f'run_{0:04d}'), record_layers)
ctrl_data_act = record_model_act(Path(EXP_DIR).joinpath('ctrl_var_target_dist_240712', f'run_{12:04d}'), record_layers)
random_data_act = record_model_act(Path(EXP_DIR).joinpath('pretrain_and_random_resnet18_0220', f'run_{0:04d}'), record_layers)
prt_data_act = record_model_act(Path(EXP_DIR).joinpath('pretrain_and_random_resnet18_0220', f'run_{5:04d}'), record_layers)

# %%
def feature_pca(run_id, record_layers, n_reduce_dims=512):
    pca_activation = {}
    save_path = Path(EXP_DIR).joinpath('ctrl_var_target_dist_240712', f'run_{run_id:04d}')
    for layer in record_layers:
        print(f'Processing {layer} PCA...')
        layer_name = layer.replace('.', '_')
        npy_pc_path = save_path.joinpath(f'act_tdw_1m_20240206_val_{layer_name}_pca_{n_reduce_dims}.npy')
        if npy_pc_path.is_file():
            d_reduced_features = np.load(npy_pc_path)
        else:
            raw_features = np.load(save_path.joinpath(f'act_tdw_1m_20240206_val_{layer_name}.npy'))
            # center raw activations before PCA
            raw_features = raw_features - raw_features.mean(axis=0)

            pca = PCA(n_components=n_reduce_dims)
            d_reduced_features = pca.fit_transform(raw_features)
            np.save(npy_pc_path, d_reduced_features)
        pca_activation[layer] = d_reduced_features
    return pca_activation

# it turned out that doing linear regression on PCAed features is much more time consuming
feature_pca(0, record_layers)
feature_pca(12, record_layers)

# %%
dset_index = dataset.dataset_index.copy()
cat_labels = [dataset.mappings['category_str2int'][wnid] for wnid in dset_index['wnid']]
dset_index['cat_labels'] = cat_labels

# %%
def validate_act(activations, record_layers, save_path):
    if save_path.is_file():
        rt_results = pd.read_csv(save_path, index_col=0)
    else:
        results = {}
        for layer in record_layers:
            print(f'Validating layer: {layer}')
            acc, std = cross_validate_on_target(activations[layer], dset_index, 
                                                'cat_labels', downsample_number=10000, num_cross_val=5, 
                                                mode='classification')
            results[layer] = [acc, std]
        rt_results = pd.DataFrame.from_dict(results, orient='index', columns=['acc', 'std'])
        rt_results.to_csv(save_path)
    return rt_results

# %%
full_data_r = validate_act(full_data_act, record_layers, Path(EXP_DIR).joinpath('ctrl_var_target_dist_240712', 'run_0000', 'cat_decoding_results.csv'))
ctrl_data_r = validate_act(ctrl_data_act, record_layers, Path(EXP_DIR).joinpath('ctrl_var_target_dist_240712', 'run_0012', 'cat_decoding_results.csv'))
random_data_r = validate_act(random_data_act, record_layers, Path(EXP_DIR).joinpath('pretrain_and_random_resnet18_0220', 'run_0000', 'cat_decoding_results.csv'))
prt_data_r = validate_act(prt_data_act, record_layers, Path(EXP_DIR).joinpath('pretrain_and_random_resnet18_0220', 'run_0005', 'cat_decoding_results.csv'))

# %%
x_axis = np.arange(len(record_layers))
fig, ax = plt.subplots(figsize=(4.8, 3.6))
ax.errorbar(x_axis - 0.1, full_data_r['acc'], yerr=full_data_r['std'], fmt='o', capsize=3, label='Full var.')
ax.errorbar(x_axis, ctrl_data_r['acc'], yerr=ctrl_data_r['std'], fmt='o', capsize=3, label='Control var.')
ax.errorbar(x_axis + 0.1, random_data_r['acc'], yerr=random_data_r['std'], fmt='o', capsize=3, label='Random')
ax.errorbar(x_axis + 0.2, prt_data_r['acc'], yerr=prt_data_r['std'], fmt='o', capsize=3, label='ImageNet1K')
ax.axhline(y=1.0/117, color='r', linestyle='--', label='Chance')
ax.set_xticks(x_axis, record_layers)
ax.set_ylabel('Category Decoding Accuracy')
ax.set_xlabel('Layer')
ax.legend()
fig.tight_layout()
fig.savefig('figures/ctrl_var_target_dist_240712_cat_decoding.pdf', transparent=True)

# %%



