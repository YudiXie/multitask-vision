import os.path as osp
import torch

NP_SEED = 1234
TCH_SEED = 2147483647

ROOT_DIR = osp.dirname(osp.abspath(__file__))
EXP_DIR = osp.join(ROOT_DIR, 'experiments')
FIG_DIR = osp.join(ROOT_DIR, 'figures')
DATA_DIR = osp.join(ROOT_DIR, 'data')

# device to run algorithm on
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')

CUDA_MODULE = 'openmind8/cuda/12.1'
CONDA_ENV = 'mtvision2'
CONDA_SCORE_ENV = 'brainscore'

print('Pytorch version: ', torch.__version__)
if USE_CUDA:
    print('Using GPU, Pytorch linked cuda version: ', torch.version.cuda)
else:
    print('Using CPU, no GPU is fund in pytorch')
