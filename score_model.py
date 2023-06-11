import os
import functools
import time

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


def prepare_model(model_identifier, load_path=None):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 78)
    model = model.to(DEVICE)
    if load_path is not None:
        model.load_state_dict(torch.load(load_path, map_location=DEVICE), strict=True)

    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    activations_model = PytorchWrapper(identifier=model_identifier, model=model, preprocessing=preprocessing)
    model = ModelCommitment(identifier=model_identifier,
                            activations_model=activations_model,
                            layers=resnet18layerlist,
                            behavioral_readout_layer='avgpool')
    
    # this run benchmark to finish layer commitment and print results
    print(f'model_name: {model.identifier}')
    print('V1 region:', model.layer_model.region_layer_map['V1'])
    print('V2 region:', model.layer_model.region_layer_map['V2'])
    print('V4 region:', model.layer_model.region_layer_map['V4'])
    print('IT region:', model.layer_model.region_layer_map['IT'])

    return model


def score_model_on_benchmarks(model, save_df, exp_group):
    for benchmark in benchmark_list:
        # The score_model will score the model on the specified benchmark.
        # When the model is asked to output activations for the IT region, it will first search for the best layer
        # and then only output this layer's activations.
        start_time = time.time()
        score = score_model(model_identifier=model.identifier, model=model,
                            benchmark_identifier=benchmark)
        print(score)
        center, error = score.sel(aggregation='center'), score.sel(aggregation='error')
        print(f"Score: {center.values:.3f}+-{error.values:.3f}")
        print("Run time %.2f mins" % ((time.time() - start_time) / 60))

        save_df = save_df.append({'model': model.identifier, 
                                  'benchmark': benchmark, 
                                  'score': center.values, 
                                  'error': error.values,
                                  'exp_group': exp_group}, ignore_index=True)
    return save_df


if __name__ == '__main__':
    save_df = pd.DataFrame(columns=['model', 'benchmark', 'score', 'error', 'exp_group'])

    # score pre-trained model
    model = prepare_model('mt0527-resnet18-pret')
    save_df = score_model_on_benchmarks(model, save_df, exp_group='Pre-trained')

    # score experiments models
    for run_id in range(10):
        load_path = os.path.join(EXP_DIR, f'multi_task_vs_categorization0527/run_000{run_id}', 'model.pth')
        model = prepare_model(model_identifier=f'mt0527-resnet18-{run_id}', 
                              load_path=load_path)
        save_df = score_model_on_benchmarks(model, save_df, exp_group='Multi-task' if run_id < 5 else 'Categorization')
            
    save_df.to_csv(os.path.join(EXP_DIR, 'multi_task_vs_categorization0527', 'mt0527_resnet18.csv'))
