import os
import functools

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from brainscore import score_model
from brainscore.benchmarks import public_benchmark_pool
# from model_tools.activations.pytorch import load_preprocess_images
# from model_tools.activations.pytorch import PytorchWrapper
# from model_tools.brain_transformation import ModelCommitment

from config_global import DEVICE

# code to get layer names
# for name, layer in model.named_modules():
#     if isinstance(layer, torch.nn.Conv2d):
#         print(name)

resnet18layerlist = [
    'conv1',
    'relu',
    'layer1.0.conv1',
    'layer1.0.conv2',
    'layer1.1.conv1',
    'layer1.1.conv2',
    'layer2.0.conv1',
    'layer2.0.conv2',
    'layer2.1.conv1',
    'layer2.1.conv2',
    'layer3.0.conv1',
    'layer3.0.conv2',
    'layer3.1.conv1',
    'layer3.1.conv2',
    'layer4.0.conv1',
    'layer4.0.conv2',
    'layer4.1.conv1',
    'layer4.1.conv2',
]

if __name__ == '__main__':

    # ImageNet pretrained model
    # model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    save_path = './experiments/multi_task_vs_categorization/run_0000'
    save_path = os.path.join(save_path, 'model.pth')

    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, 78)
    model.load_state_dict(torch.load(save_path, map_location=DEVICE), strict=True)

    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    activations_model = PytorchWrapper(identifier='my-model', model=model, preprocessing=preprocessing)

    model = ModelCommitment(identifier='my-model',
                            activations_model=activations_model,
                            # specify layers to consider
                            layers=resnet18layerlist)
    
    # The score_model will score the model on the specified benchmark.
    # When the model is asked to output activations for the IT region, it will first search for the best layer
    # and then only output this layer's activations.
    score = score_model(model_identifier=model.identifier, model=model,
                        benchmark_identifier='dicarlo.MajajHong2015public.IT-pls')
    print(score)
    