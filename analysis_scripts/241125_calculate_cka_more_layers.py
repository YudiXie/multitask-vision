# %%
from pathlib import Path
import numpy as np
import pandas as pd

import torchvision.transforms as transforms
from tqdm import trange

from dataset import TDWDataset
from utils import prepare_pytorch_model
from activity import get_model_activations
from config_global import EXP_DIR, DATA_DIR
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

# %%
# extract and save model activations
for run_id in range(80):
    get_model_act(Path(EXP_DIR).joinpath('multi_task_tdw_1m20240206_0718', f'run_{run_id:04d}'), dataset, record_layers, 674)
for run_id in range(6):
    get_model_act(Path(EXP_DIR).joinpath('pretrain_and_random_resnet18_0220', f'run_{run_id:04d}'), dataset, record_layers, 674)
for run_id in range(8):
    get_model_act(Path(EXP_DIR).joinpath('imagenet1k_0902', f'run_{run_id:04d}'), dataset, record_layers, 1000)


# %%
tasks_maps = {'Dis. reg.': ('multi_task_tdw_1m20240206_0718', [i for i in range(8)]),
              'Tra. reg.': ('multi_task_tdw_1m20240206_0718', [i for i in range(8, 16)]),
              'Rot reg.': ('multi_task_tdw_1m20240206_0718', [i for i in range(16, 24)]),
              'Dis. Tra. Rot.': ('multi_task_tdw_1m20240206_0718', [i for i in range(48, 56)]),
              'Cat. cla.': ('multi_task_tdw_1m20240206_0718', [i for i in range(56, 64)]),
              'All cla. all reg.': ('multi_task_tdw_1m20240206_0718', [i for i in range(72, 80)]),
              'ImageNet cla.': ('imagenet1k_0902', [i for i in range(8)]),
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



