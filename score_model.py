import os
import functools
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from brainscore import score_model
from model_tools.activations.pytorch import load_preprocess_images
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.brain_transformation import ModelCommitment

from config_global import DEVICE, EXP_DIR
from train import log_complete
from utils import load_config

# code to get layer names
# for name, layer in model.named_modules():
#     if isinstance(layer, torch.nn.Conv2d):
#         print(name)

resnet18layerlist = [
    'relu',
    'layer1.0.relu',
    'layer1.1.relu',
    'layer2.0.relu',
    'layer2.1.relu',
    'layer3.0.relu',
    'layer3.1.relu',
    'layer4.0.relu',
    'layer4.1.relu',
    'avgpool',
    'fc',
]


benchmark_list = [
    'movshon.FreemanZiemba2013public.V1-pls',
    'movshon.FreemanZiemba2013public.V2-pls',
    'dicarlo.MajajHong2015public.V4-pls',
    'dicarlo.MajajHong2015public.IT-pls',
    'dicarlo.Rajalingham2018public-i2n',
    ]


def get_layer_commitment(model: ModelCommitment):
    """
    at first run, this run benchmark to finish layer commitment and print results 
    subsequent runs will load the layer commitment from the saved file
    args:
        model: ModelCommitment object
    """
    print(f'model_identifier: {model.identifier}')
    print('V1 region:', model.layer_model.region_layer_map['V1'])
    print('V2 region:', model.layer_model.region_layer_map['V2'])
    print('V4 region:', model.layer_model.region_layer_map['V4'])
    print('IT region:', model.layer_model.region_layer_map['IT'])


def prepare_model(model_identifier: str, load_path: str = '') -> ModelCommitment:
    """
    prepare model for benchmarking
    args:
        model_identifier: str, unique model identifier for benchmarking
        load_path: str, path to load model weights, 
            if provided load weights, otherwise use pretrained weights
    return:
        model: ModelCommitment object
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 78)
    model = model.to(DEVICE)
    if load_path != '':
        print(f'loading model from {load_path}')
        model.load_state_dict(torch.load(load_path, map_location=DEVICE), strict=True)

    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    activations_model = PytorchWrapper(identifier=model_identifier, model=model, preprocessing=preprocessing)
    model = ModelCommitment(identifier=model_identifier,
                            activations_model=activations_model,
                            layers=resnet18layerlist,
                            behavioral_readout_layer='avgpool')
    return model


def score_model_on_a_benchmark(model: ModelCommitment, 
                               benchmark: str,
                               log_path: str = ''):
    """
    score model on a benchmark
    args:
        model: ModelCommitment object
        benchmark: str, benchmark name
        log_path: str, path to save log file, if provided, otherwise not save
    return:
        score: xarray DataArray, the model score on that benchmark
    """
    start_time = datetime.now()

    score = score_model(model_identifier=model.identifier, model=model,
                        benchmark_identifier=benchmark)
    print(score)
    center, error = score.sel(aggregation='center'), score.sel(aggregation='error')
    print(f"Score: {center.values:.3f}+-{error.values:.3f}")

    complete_time = datetime.now()
    print(f'Scoring time: {str(complete_time - start_time)}')
    if log_path != '':
        log_complete(log_path, start_time, 'score')
    return score


def prepare_and_score_model(config):
    """
    prepare and score model on all benchmarks
    args:
        config: dict, an experimental config specifying a model
    """
    model_id = '-'.join([config['experiment_name'], 
                         config['model_archi'], str(config['run_id'])])
    model_save_path = os.path.join(config['save_path'], 'model.pth')
    
    model = prepare_model(model_id, model_save_path)
    for benchmark in benchmark_list:
        score_model_on_a_benchmark(model, benchmark, config['save_path'])


def prepare_and_score_model_slurm(config_path):
    config = load_config(config_path)
    prepare_and_score_model(config)


def score_model_on_benchmarks_save(model: ModelCommitment, save_df, exp_group):
    for benchmark in benchmark_list:
        score = score_model_on_a_benchmark(model, benchmark)
        save_df = save_df.append({'model': model.identifier, 
                                  'benchmark': benchmark, 
                                  'score': score.sel(aggregation='center').values, 
                                  'error': score.sel(aggregation='error').values,
                                  'exp_group': exp_group}, ignore_index=True)
    return save_df


if __name__ == '__main__':
    exp_name = 'multi_task_0610'
    number_runs = 30

    save_df = pd.DataFrame(columns=['model', 'benchmark', 'score', 'error', 'exp_group'])
    
    # score pre-trained model
    model = prepare_model(f'{exp_name}-resnet18-pret')
    get_layer_commitment(model)
    save_df = score_model_on_benchmarks_save(model, save_df, exp_group='Pre-trained')

    # score experiments models
    for run_id in range(number_runs):

        if run_id < 5:
            exp_group = 'Multi-task'
        elif run_id < 10:
            exp_group = 'Categorization'
        elif run_id < 15:
            exp_group = 'Multi_task_wo_object_class'
        elif run_id < 20:
            exp_group = 'Size_reg'
        elif run_id < 25:
            exp_group = 'Translation_reg'
        else:
            exp_group = 'Rotation_reg'

        load_path = os.path.join(EXP_DIR, f'{exp_name}', f'run_{run_id:04d}', 'model.pth')
        model = prepare_model(model_identifier=f'{exp_name}-resnet18-{run_id}', 
                              load_path=load_path)
        get_layer_commitment(model)
        save_df = score_model_on_benchmarks_save(model, save_df, exp_group=exp_group)
        
            
    save_df.to_csv(os.path.join(EXP_DIR, exp_name, 'resnet18_brainscore_results.csv'))
