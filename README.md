# multitask-vision
A repo for training vision models with multiple tasks

### Installing environment:

```
conda create -n multitask-vision python
conda activate multitask-vision
# install pytorch, on linux it is, might be different on MacOS.
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyyaml scikit-image pandas
conda install -c conda-forge tqdm

# install weights and biases
conda install wandb --channel conda-forge
# or
pip install wandb

# then login to wandb
wandb login
```

### Using the code:
First, find the experiment to run or write new experimental conditions in `exp_config_list.py`

Then, prepare the datset, by running:
```
python dataset.py -n <dataset-name>
```

Train models, this will trained a group of neural network models with the dataset specified in `exp_config_list.py`

Use `-c`, to run on the openmind cluster, this will submit many jobs, each for the training or scoring for one neural network model.

```
python main.py -d train -n <experiment-name> -c
```

After the training is complete, check if all the train runs are finished successfully.

Use `-m`, to check for any runs that are not finished.

```
python main.py -d train -n <experiment-name> -c -m
```

Score models, this will calculate the Brain-Score for the group of neural network models.
```
python main.py -d score -n <experiment-name> -c
```

After the scoring is complete, check if all the score runs are finished successfully.

```
python main.py -d score -n <experiment-name> -c -m
```

Run the following to complie the scores into a consolidated csv file

```
python score_model.py -n <experiment-name>
```

## examples:
```
python main.py -d train -n multi_task_tdw_large20230907_nopret_0925 -c -g a100 -t 12
```