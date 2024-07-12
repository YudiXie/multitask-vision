# multitask-vision
A repo for training vision models with multiple tasks

### Installing environment:

```bash
conda create -n mtvision2 python=3.10
conda activate mtvision2

conda install ipykernel pyyaml scikit-image pandas tqdm matplotlib

# install pytorch, on linux with cuda 11.7
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
# or on MacOS
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 -c pytorch

# install weights and biases
conda install wandb --channel conda-forge
# or
pip install wandb

# then login to wandb
wandb login
```

To analyze model neural alignment, you would need to install Brain-Score as well. See [here](https://github.com/brain-score/brain-score).

### Using the code:
First, find the experiment to run or write new experimental conditions in `exp_config_list.py`

Then, prepare the datset, by running:
```bash
python dataset.py -n <dataset-name>
```

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

Preprocess the dataset for use
```bash
python tdw_dataset_preprocess.py --index <path-to-index>
```

## Examples:
```
python main.py -d train -n multi_task_tdw_large20230907_nopret_0925 -c -g a100 -t 12
```
