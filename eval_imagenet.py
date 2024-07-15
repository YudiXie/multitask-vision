import os
from datetime import datetime
from pathlib import Path

import torch.nn as nn

from transferscore.imagenet_acc import get_model_imagnet_acc

from config_global import DEVICE
from utils import load_config, log_complete, prepare_pytorch_model
from tasks_setup import get_output_info


def eval_model_imagenet(config):
    """
    prepare and score model on all benchmarks
    args:
        config: dict, an experimental config specifying a model
    """
    model_save_path = os.path.join(config['save_path'], 'model.pth')
    out_dim, _ignore = get_output_info(config['dataset_name'])
    model = prepare_pytorch_model(config['model_archi'], out_dim, model_save_path)
    model.fc = nn.Linear(model.fc.in_features, 1000)
    model = model.to(DEVICE)

    start_time = datetime.now()
    get_model_imagnet_acc(model, '/om/user/yu_xie/data/ImageNet', 
                          save_path=Path(config['save_path']).joinpath('imagenet_acc.csv'))
    complete_time = datetime.now()
    print(f'ImageNet eval time: {str(complete_time - start_time)}')
    log_complete(config['save_path'], start_time, 'imneval')


def eval_model_imagenet_slurm(config_path):
    config = load_config(config_path)
    eval_model_imagenet(config)
