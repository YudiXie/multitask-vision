import os
from datetime import datetime

import yaml
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from config_global import DEVICE


def log_complete(exp_path: str, start_time=None, mode='train'):
    """
    create a file to indicate the operation is finished
    operation could be 'train' or 'score' or many others
    args:
        exp_path: str, a directory path to save the log file
        start_time: datetime, the start time of the operation
            if None, only the complete time will be logged
        mode: str, the operation name, will be saved as the log file name
            and printed in the terminal and log file
    """
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    
    complete_time = datetime.now()
    with open(os.path.join(exp_path, f'{mode}_complete.txt'), 'w') as f:
        f.write(f'{mode} is complete at: {complete_time.strftime("%Y-%m-%d %H:%M:%S")}')
        if start_time is not None:
            f.write(f'\n{mode} time: {str(complete_time - start_time)}')
    
    print(f'{mode} is complete at: {complete_time.strftime("%Y-%m-%d %H:%M:%S")}')


def save_config(config, save_folder='./exp_configs'):
    """
    save a config dictrionary to a yaml file in the folder
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    yaml_file_path = os.path.join(save_folder, 'config.yml')
    with open(yaml_file_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    return yaml_file_path


def load_config(yaml_file_path='config.yml'):
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def prepare_pytorch_model(out_dim: int, load_path: str = ''):
    """
    prepare a torch.nn model
    args:
        out_dim: int, the output dimension of the model
        load_path: str, path to load model weights, 
            if provided load weights, otherwise use pretrained weights
    return:
        model: torch.nn model
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, out_dim)
    model = model.to(DEVICE)
    
    # load model from saved weights
    if load_path:
        model.load_state_dict(torch.load(load_path, map_location=DEVICE), strict=True)
        print(f'Loaded model from {load_path}')
    else:
        print(f'Loaded model from pretrained weights')
    
    return model


def find_region_layer(df, region, model_id):
    """
    find the layer name for a model for a particular benchmark region
    """
    layer_series = df[(df['model'] == model_id) & (df['benchmark_region'] == region)]['mapped_layer']
    assert len(layer_series) == 1
    return layer_series.to_numpy(copy=True)[0]


def get_model_id(config):
    """
    get a unique model id from the config dictionary
    """
    return '-'.join([config['experiment_name'], 
                     config['model_archi'], 
                     str(config['run_id'])])


def cosine_sim(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))


def abs_cosine_sim(A, B):
    return np.abs(cosine_sim(A, B))


def trim_diagonal(mat):
    assert mat.shape[0] == mat.shape[1]  # check if the matrix is square
    n = mat.shape[0]  # get the size of the matrix
    trimmed_mat = np.zeros((n, n - 1))  # initialize the trimmed matrix
    
    for i in range(n):
        trimmed_mat[i, :i] = mat[i, :i]  # elements before the diagonal
        trimmed_mat[i, i:] = mat[i, i+1:]  # elements after the diagonal
    
    return trimmed_mat
