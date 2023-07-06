import os
import functools
import time
from datetime import datetime
import argparse

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
from utils import load_config, log_complete
import exp_config_list

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


benchmark_dict = {
    'V1': 'movshon.FreemanZiemba2013public.V1-pls',
    'V2': 'movshon.FreemanZiemba2013public.V2-pls',
    'V4': 'dicarlo.MajajHong2015public.V4-pls',
    'IT': 'dicarlo.MajajHong2015public.IT-pls',
    'Behavior': 'dicarlo.Rajalingham2018public-i2n',
    }


def get_layer_commitment(model: ModelCommitment):
    """
    at first run, this run benchmark to finish layer commitment and print results 
    subsequent runs will load the layer commitment from the saved file
    args:
        model: ModelCommitment object
    """
    layer_map = {}
    layer_map['V1'] = model.layer_model.region_layer_map['V1']
    layer_map['V2'] = model.layer_model.region_layer_map['V2']
    layer_map['V4'] = model.layer_model.region_layer_map['V4']
    layer_map['IT'] = model.layer_model.region_layer_map['IT']
    # layer_map['Behavior']
    return layer_map


def prepare_pytorch_model(load_path: str = ''):
    """
    prepare a torch.nn model
    args:
        load_path: str, path to load model weights, 
            if provided load weights, otherwise use pretrained weights
    return:
        model: torch.nn model
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 78)
    model = model.to(DEVICE)
    
    # load model from saved weights
    if load_path != '':
        print(f'loading model from {load_path}')
        model.load_state_dict(torch.load(load_path, map_location=DEVICE), strict=True)
    
    return model


def prepare_model_commitment(model_identifier: str, load_path: str = '') -> ModelCommitment:
    """
    prepare model for benchmarking
    args:
        model_identifier: str, unique model identifier for benchmarking
        load_path: str, path to load model weights, 
            if provided load weights, otherwise use pretrained weights
    return:
        model: ModelCommitment object
    """
    pytorch_model = prepare_pytorch_model(load_path)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    activations_model = PytorchWrapper(identifier=model_identifier, 
                                       model=pytorch_model,
                                       preprocessing=preprocessing)
    model_commitment = ModelCommitment(identifier=model_identifier,
                                       activations_model=activations_model,
                                       layers=resnet18layerlist,
                                       behavioral_readout_layer='avgpool')
    return model_commitment


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
    print(f'Scoring time for {benchmark}: {str(complete_time - start_time)}')
    if log_path != '':
        log_complete(log_path, start_time, f'score_{benchmark}')
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
    
    model = prepare_model_commitment(model_id, model_save_path)

    start_time = datetime.now()
    for region, benchmark_id in benchmark_dict.items():
        score_model_on_a_benchmark(model, benchmark_id)
    
    complete_time = datetime.now()
    print(f'Scoring time for all benchmarks: {str(complete_time - start_time)}')
    log_complete(config['save_path'], start_time, 'score')


def prepare_and_score_model_slurm(config_path):
    config = load_config(config_path)
    prepare_and_score_model(config)


def save_model_scores(model: ModelCommitment, save_df, exp_group):
    """
    read model score on benchmarks and append to save_df
    """
    layer_map = get_layer_commitment(model)
    layer_map['Behavior'] = 'avgpool'
    for region, benchmark_id in benchmark_dict.items():
        score = score_model_on_a_benchmark(model, benchmark_id)
        save_df = save_df.append({'model': model.identifier, 
                                  'benchmark_region': region, 
                                  'benchmark_id': benchmark_id,
                                  'mapped_layer': layer_map[region],
                                  'score': score.sel(aggregation='center').values, 
                                  'error': score.sel(aggregation='error').values,
                                  'exp_group': exp_group,
                                  }, 
                                  ignore_index=True)
    return save_df


def save_exp_scores(exp_name):
    """
    save model scores for all models in an experiment
    alone with pre-trained models
    to brainscore_results.csv in the experiment folder
    """
    config_list = getattr(exp_config_list, exp_name)()

    save_df = pd.DataFrame(columns=['model',
                                    'benchmark_region',
                                    'benchmark_id',
                                    'mapped_layer',
                                    'score',
                                    'error',
                                    'exp_group',
                                    ])
    
    # save score for pre-trained model
    model = prepare_model_commitment('mt0527-resnet18-pret')
    save_df = save_model_scores(model, save_df, exp_group='Pre-trained')

    # save score for models specified by experiment config list
    for config in config_list:
        model_id = '-'.join([config['experiment_name'], 
                             config['model_archi'], str(config['run_id'])])
        model_save_path = os.path.join(config['save_path'], 'model.pth')

        model = prepare_model_commitment(model_identifier=model_id, load_path=model_save_path)
        save_df = save_model_scores(model, save_df, exp_group=config['group_name'])
    
    save_df.to_csv(os.path.join(EXP_DIR, config_list[0]['experiment_name'], 'brainscore_results.csv'))


if __name__ == '__main__':
    # run this script to save model scores for all models in an experiment
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', help='Name of the experiment')
    args = parser.parse_args()

    save_exp_scores(args.name)
