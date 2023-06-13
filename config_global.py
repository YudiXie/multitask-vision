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
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

CUDA_MODULE = 'openmind8/cuda/11.7'
CONDA_ENV = 'multitask-vision'
CONDA_SCORE_ENV = 'brainscore'
