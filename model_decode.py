from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from sklearn.random_projection import johnson_lindenstrauss_min_dim

from dataset import TDWDataset
from activity import cross_validate_on_target, get_model_activations
from utils import prepare_pytorch_model, load_config, log_complete


# model must be a resnet
def remove_resnet_duplicates(activity_dict):
    # reduce the duplicate activations in resnet
    # because the later relu layer are used twice in resnet,
    for k, v in activity_dict.items():
        if '.relu' in k:
            v.pop(-2)


def get_model_act(run_path, dataset, record_layers):
    """
    record the activations of the model on the dataset on the specified layers
    :param run_path: the path to save the activations
    :param dataset: pytorch dataset object, the dataset to record the activations
    :param record_layers: the layers to record
    """
    npy_path_dict = {}
    for layer in record_layers:
        layer_name = layer.replace('.', '_')
        npy_path_dict[layer] = run_path.joinpath(f'act_{layer_name}_{dataset.dset_name}.npy')

    has_records = [True if npy_path.is_file() else False for npy_path in npy_path_dict.values()]
    
    if all(has_records):
        print('All activations are already recorded!')
        all_activations = {}
        for layer, npy_path in npy_path_dict.items():
            all_activations[layer] = np.load(npy_path)
        return all_activations
    else:
        print('Start recording activations ...')
        model = prepare_pytorch_model('resnet18', 674, run_path.joinpath('model.pth'))
        all_activations = get_model_activations(dataset, model, record_layers, remove_resnet_duplicates)
        for layer, npy_path in npy_path_dict.items():
            if npy_path.is_file():
                npy_path.unlink()
            np.save(npy_path, all_activations[layer])
        return all_activations


def validate_act(activations, data_index, record_layers, save_path):
    """
    Validate the activations on the dataset
    :param activations: dict, the activations of the model on the dataset, with keys as layer names in record_layers
    :param data_index: pd.DataFrame, the index of the dataset
    :param record_layers: list, the layers to validate
    :param save_path: str, the path to save the validation results
    """
    if save_path.is_file():
        save_path.unlink()

    results = {}
    for layer in record_layers:
        print(f'Validating layer: {layer}')
        
        layer_act = activations[layer]
        random_red_dim = johnson_lindenstrauss_min_dim(layer_act.shape[0])
        if layer_act.shape[1] > random_red_dim:
            # use random projection to downsample, dimension is determined by the Johnson-Lindenstrauss lemma
            # eg. 2000 samples -> 6515 dims
            reduce_met = 'random'
            reduce_dim = random_red_dim
        else:
            # do not downsample
            reduce_met = 'none'
            reduce_dim = None
        results[layer] = cross_validate_on_target(activations[layer], data_index, 'cat_labels',
                                                  downsample_method=reduce_met,
                                                  downsample_number=reduce_dim,
                                                  num_cross_val=5,
                                                  mode='classification')
    rt_results = pd.DataFrame.from_dict(results)
    rt_results.to_csv(save_path)
    return rt_results


def decode_from_model(config):
    start_time = datetime.now()

    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    dataset = TDWDataset(root_dir='/om/user/yu_xie/data/tdw_images/tdw_1m_20240206',
                        split='val', fraction=0.04, transform=transform)
    record_layers = ['layer1.0.relu', 'layer2.0.relu', 'layer3.0.relu', 'layer4.0.relu', 'avgpool']

    dset_index = dataset.dataset_index.copy()
    cat_labels = [dataset.mappings['category_str2int'][wnid] for wnid in dset_index['wnid']]
    dset_index['cat_labels'] = cat_labels

    run_path = Path(config['save_path'])
    model_act = get_model_act(run_path, dataset, record_layers)
    validate_act(model_act, dset_index, record_layers, run_path.joinpath('cat_decoding_results_240820.csv'))
    
    complete_time = datetime.now()
    print(f'Decoding completed! total time: {str(complete_time - start_time)}')
    log_complete(config['save_path'], start_time, 'decode')


def decode_from_model_slurm(config_path):
    config = load_config(config_path)
    decode_from_model(config)
