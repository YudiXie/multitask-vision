# %%
import os
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from config_global import DEVICE, EXP_DIR

# %%
for run_id in range(5):
    model = resnet50()
    model.fc = nn.Linear(model.fc.in_features, 674)
    model = model.to(DEVICE)
    exp_path = os.path.join(EXP_DIR, 'pretrain_and_random_resnet50_0220', f'run_{run_id:04d}')
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    torch.save(model.state_dict(), 
               os.path.join(EXP_DIR, 'pretrain_and_random_resnet50_0220', 
                            f'run_{run_id:04d}', 'model.pth'))

# %%
for run_id in range(5, 10):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 674)
    model = model.to(DEVICE)
    exp_path = os.path.join(EXP_DIR, 'pretrain_and_random_resnet50_0220', f'run_{run_id:04d}')
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    torch.save(model.state_dict(), 
               os.path.join(EXP_DIR, 'pretrain_and_random_resnet50_0220', 
                            f'run_{run_id:04d}', 'model.pth'))


