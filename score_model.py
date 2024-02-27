import os
import sys
import functools
from datetime import datetime
import argparse

import pandas as pd

from brainscore_vision import load_benchmark
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images, PytorchWrapper

from config_global import EXP_DIR, DATA_DIR
from utils import load_config, log_complete, prepare_pytorch_model, get_model_id
import exp_config_list
from tasks_setup import get_output_info
from pathlib import Path

# code to get layer names
# for name, layer in model.named_modules():
#     if isinstance(layer, torch.nn.Conv2d):
#         print(name)

resnet18_scorelayerlist = [
    'relu',
    'layer1.0.relu',
    'layer1.1.relu',
    'layer2.0.relu',
    'layer2.1.relu',
    'layer3.0.relu',
    'layer3.1.relu',
    'layer4.0.relu',
    'layer4.1.relu',
    'avgpool',
    'fc',
]

resnet50_scorelayerlist = [
    'relu',
    'layer1.0.relu',
    'layer1.1.relu',
    'layer1.2.relu',
    'layer2.0.relu',
    'layer2.1.relu',
    'layer2.2.relu',
    'layer2.3.relu',
    'layer3.0.relu',
    'layer3.1.relu',
    'layer3.2.relu',
    'layer3.3.relu',
    'layer3.4.relu',
    'layer3.5.relu',
    'layer4.0.relu',
    'layer4.1.relu',
    'layer4.2.relu',
    'avgpool',
    'fc',
]

score_layers = {
    'resnet18': resnet18_scorelayerlist,
    'resnet50': resnet50_scorelayerlist,
}


benchmark_dict = {
    'V1': 'FreemanZiemba2013public.V1-pls',
    'V2': 'FreemanZiemba2013public.V2-pls',
    'V4': 'MajajHong2015public.V4-pls',
    'IT': 'MajajHong2015public.IT-pls',
    'Behavior': 'Rajalingham2018public-i2n',
    }


def score_local_model(model, benchmark_identifier):
    """
    Score a brain model on the benchmark referenced by the `benchmark_identifier`.
    args:
        model: BrainModel (ModelCommitment) object
        benchmark_identifier: str, unique benchmark identifier
    """ 
    benchmark = load_benchmark(benchmark_identifier)
    score = benchmark(model)
    score.attrs['model_identifier'] = model.identifier
    score.attrs['benchmark_identifier'] = benchmark_identifier
    try:  # attempt to look up the layer commitment if model uses a standard layer model
        score.attrs['comment'] = f"layers: {model.layer_model.region_layer_map}"
    except Exception:
        pass
    return score


def get_layer_commitment(model: ModelCommitment):
    """
    at first run, this run benchmark to finish layer commitment and print results 
    subsequent runs will load the layer commitment from the saved file
    args:
        model: ModelCommitment object
    """
    layer_map = {}
    layer_map['V1'] = model.layer_model.region_layer_map['V1']
    layer_map['V2'] = model.layer_model.region_layer_map['V2']
    layer_map['V4'] = model.layer_model.region_layer_map['V4']
    layer_map['IT'] = model.layer_model.region_layer_map['IT']
    # layer_map['Behavior']
    return layer_map


def prepare_model_commitment(model_archi: str, model_identifier: str, 
                             out_dim: int, load_path: str = '') -> ModelCommitment:
    """
    prepare model for benchmarking
    args:
        model_archi: str, model architecture name
        model_identifier: str, unique model identifier for benchmarking
        load_path: str, path to load model weights, 
            if provided load weights, otherwise use pretrained weights
    return:
        model: ModelCommitment object
    """
    pytorch_model = prepare_pytorch_model(model_archi, out_dim, load_path)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    activations_model = PytorchWrapper(identifier=model_identifier, 
                                       model=pytorch_model,
                                       preprocessing=preprocessing)
    model_commitment = ModelCommitment(identifier=model_identifier,
                                       activations_model=activations_model,
                                       layers=score_layers[model_archi],
                                       behavioral_readout_layer='avgpool')
    return model_commitment


def score_model_on_a_benchmark(model: ModelCommitment, 
                               benchmark: str,
                               log_path: str = ''):
    """
    score model on a benchmark, and save the results
    args:
        model: ModelCommitment object
        benchmark: str, benchmark name
        log_path: str, path to save log file, if provided, otherwise not save
    return:
        score: the model score on that benchmark
        error: the error of the score
    """
    start_time = datetime.now()

    score_path = Path(DATA_DIR).joinpath(f'{model.identifier}_{benchmark}_score.csv')
    if score_path.is_file():
        read_df = pd.read_csv(score_path, index_col=0)
        score, error = read_df['score'][0], read_df['error'][0]
    else:
        bscore = score_local_model(model=model, benchmark_identifier=benchmark)
        score, error = bscore.item(), bscore.error.item()
        save_dict = {'score': [score, ], 'error': [error, ]}
        pd.DataFrame.from_dict(save_dict).to_csv(score_path)
    
    complete_time = datetime.now()
    print(f"{model.identifier} on {benchmark} score: {score:.3f} +- {error:.3f}")
    print(f'Scoring time: {str(complete_time - start_time)}')
    if log_path != '':
        log_complete(log_path, start_time, f'score_{benchmark}')
    return score, error


def prepare_and_score_model(config):
    """
    prepare and score model on all benchmarks
    args:
        config: dict, an experimental config specifying a model
    """
    model_save_path = os.path.join(config['save_path'], 'model.pth')
    out_dim, _ignore = get_output_info(config['dataset_name'])
    model = prepare_model_commitment(config['model_archi'], get_model_id(config), 
                                     out_dim, model_save_path)

    start_time = datetime.now()
    for region, benchmark_id in benchmark_dict.items():
        score_model_on_a_benchmark(model, benchmark_id)
    
    complete_time = datetime.now()
    print(f'Scoring time for all benchmarks: {str(complete_time - start_time)}')
    log_complete(config['save_path'], start_time, 'score')


def prepare_and_score_model_slurm(config_path):
    config = load_config(config_path)
    prepare_and_score_model(config)


def save_model_scores(model: ModelCommitment, save_df, exp_group):
    """
    read model score on benchmarks and append to save_df
    """
    for region, benchmark_id in benchmark_dict.items():
        score_path = Path(DATA_DIR).joinpath(f'{model.identifier}_{benchmark_id}_score.csv')
        if score_path.is_file():
            layer_map = get_layer_commitment(model)
            layer_map['Behavior'] = 'avgpool'

            read_df = pd.read_csv(score_path, index_col=0)
            score, error = read_df['score'][0], read_df['error'][0]
            save_df = save_df.append({'model': model.identifier, 
                                    'benchmark_region': region, 
                                    'benchmark_id': benchmark_id,
                                    'mapped_layer': layer_map[region],
                                    'score': score, 
                                    'error': error,
                                    'exp_group': exp_group,
                                    }, 
                                    ignore_index=True)
        else:
            print(f'No score file found for {model.identifier} on {benchmark_id}')
            if input('Continue? (yes/no): ') != 'yes':
                sys.exit("exit program.")
    return save_df


def save_exp_scores(exp_name):
    """
    save model scores for all models in an experiment
    alone with pre-trained models
    to brainscore_results.csv in the experiment folder

    need to do: the layer committement is not saved intuitively
    so that the need to call the score_model_on_a_benchmark and prepare model
    future version should save the layer commitment together with score
    """
    config_list = getattr(exp_config_list, exp_name)()

    save_df = pd.DataFrame(columns=['model',
                                    'benchmark_region',
                                    'benchmark_id',
                                    'mapped_layer',
                                    'score',
                                    'error',
                                    'exp_group',
                                    ])

    # save score for models specified by experiment config list
    for config in config_list:
        model_save_path = os.path.join(config['save_path'], 'model.pth')
        out_dim, _ignore = get_output_info(config['dataset_name'])
        model = prepare_model_commitment(config['model_archi'], get_model_id(config),
                                         out_dim, model_save_path)
        save_df = save_model_scores(model, save_df, exp_group=config['group_name'])
    
    save_df.to_csv(os.path.join(EXP_DIR, config_list[0]['experiment_name'], 'brainscore_results.csv'))


if __name__ == '__main__':
    # run this script to save model scores for all models in an experiment
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', help='Name of the experiment')
    args = parser.parse_args()

    save_exp_scores(args.name)
