from activity import get_activity_on_dataset, evaluate_regression
from torchvision.models import resnet18, ResNet18_Weights, ResNet
import os
from config_global import DEVICE, EXP_DIR
import torch.nn as nn
import torch

from matplotlib import pyplot as plt


if __name__ == '__main__':

    def prepare_model(run_id):
        exp_name = 'multi_task_0610'
        load_path = os.path.join(EXP_DIR, f'{exp_name}', f'run_{run_id:04d}', 'model.pth')
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 78)
        model = model.to(DEVICE)
        print(f'loading model from {load_path}')
        model.load_state_dict(torch.load(load_path, map_location=DEVICE), strict=True)
        return model


    def get_correlation(layers, data):
        corr_list = []
        for layer in layers:
            corr_list.append(evaluate_regression(layer, 's', **data)[0])
        return corr_list

    cat_model = prepare_model(5)
    mul_model = prepare_model(10)

    layer_list = ['layer3.1.relu', 'avgpool']
    cat_data = get_activity_on_dataset(cat_model, layer_list)
    mul_data = get_activity_on_dataset(mul_model, layer_list)


    layers = ['layer3.1.relu', 'avgpool']
    cat_corrs = get_correlation(layers, cat_data)
    mul_corrs = get_correlation(layers, mul_data)

    plt.plot(cat_corrs, label='cat')
    plt.plot(mul_corrs, label='mul')
    plt.legend()
    plt.show()
