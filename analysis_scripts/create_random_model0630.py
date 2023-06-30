import os
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from config_global import DEVICE, EXP_DIR

for run_id in range(5):
    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, 78)
    model = model.to(DEVICE)
    exp_path = os.path.join(EXP_DIR, 'random_models0630', f'run_{run_id:04d}')
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    torch.save(model.state_dict(), 
               os.path.join(EXP_DIR, 'random_models0630', 
                            f'run_{run_id:04d}', 'model.pth'))
