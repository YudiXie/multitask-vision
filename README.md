# multitask-vision
A repo for training vision models with multiple tasks

Installing environment:
"""
# not tested
conda create -n multitask-vision python
conda activate multitask-vision
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyyaml scikit-image pandas

conda install wandb --channel conda-forge
# or
pip install wandb

# then wandb login
wandb login
"""

