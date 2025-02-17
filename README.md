# multitask-vision
Code for the paper: [Xie, Y., Huang, W., Alter, E., Schwartz, J., Tenenbaum, J.B. and DiCarlo, J.J., 2024. Vision CNNs trained to estimate spatial latents learned similar ventral-stream-aligned representations. arXiv preprint arXiv:2412.09115.](https://arxiv.org/abs/2412.09115)

Link to code to generate dataset: [https://github.com/YudiXie/tdw_image_dataset](https://github.com/YudiXie/tdw_image_dataset)

### Installing environment:

```bash
conda create -n mtvision3 python=3.10
conda activate mtvision3

conda install ipykernel pyyaml scikit-image pandas tqdm matplotlib

# install pytorch, on linux with cuda
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
# or on MacOS
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 -c pytorch

# install weights and biases
pip install wandb

# then login to wandb
wandb login

# to evaluate model imagenet performance, install transfer score
git clone https://github.com/YudiXie/transfer-score.git
pip install -e . --config-settings editable_mode=compat

# to make plots, install easyfigs
git clone https://github.com/YudiXie/easyfigs.git
pip install -e . --config-settings editable_mode=compat

# install rsatoolbox
pip install rsatoolbox
```

To analyze model neural alignment, you would need to install Brain-Score as well. See [here](https://github.com/brain-score/brain-score).

### Using the code:
Preprocess the dataset for use. This will generate a shuffled dataset index for training and testing, create a mapping between label names and label indices, and calculate the mean and std of some of the targets, so that they can be used to normalize them during training and testing.
```bash
python tdw_dataset_preprocess.py --index <path-to-index>
```

First, find the experiment to run or write new experimental conditions in `exp_config_list.py`

Train models, this will trained a group of neural network models with the dataset specified in `exp_config_list.py`

Use `-c`, to run on the openmind cluster, this will submit many jobs, each for the training or scoring for one neural network model.

```bash
python main.py -d train -n <experiment-name> -c
```

After the training is complete, check if all the train runs are finished successfully.

Use `-m`, to check for any runs that are not finished.

```bash
python main.py -d train -n <experiment-name> -c -m
```

Score models, this will calculate the Brain-Score for the group of neural network models.
```bash
python main.py -d score -n <experiment-name> -c
```

After the scoring is complete, check if all the score runs are finished successfully.

```bash
python main.py -d score -n <experiment-name> -c -m
```

Run the following to complie the scores into a consolidated csv file

```bash
python score_model.py -n <experiment-name>
```

## Examples:
```
python main.py -d train -n multi_task_tdw_large20230907_nopret_0925 -c -g a100 -t 12
```
