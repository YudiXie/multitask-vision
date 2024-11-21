import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from cka import linear_CKA
from config_global import EXP_DIR, DATA_DIR


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--do', type=int, help='index to run')
    args = parser.parse_args()

    idx1 = args.do

    tasks_maps = {
        'Dis. reg.': ('multi_task_tdw_1m20240206_0718', [i for i in range(8)]),
        'Tra. reg.': ('multi_task_tdw_1m20240206_0718', [i for i in range(8, 16)]),
        'Rot reg.': ('multi_task_tdw_1m20240206_0718', [i for i in range(16, 24)]),
        'Dis. Tra. Rot.': ('multi_task_tdw_1m20240206_0718', [i for i in range(48, 56)]),
        'Cat. cla.': ('multi_task_tdw_1m20240206_0718', [i for i in range(56, 64)]),
        'All cla. all reg.': ('multi_task_tdw_1m20240206_0718', [i for i in range(72, 80)]),
        'ImageNet cla.': ('imagenet1k_0902', [i for i in range(8)]),
        'Untrained': ('pretrain_and_random_resnet18_0220', [i for i in range(5)]),
    }
    
    task_name_list = []
    exp_name_list = []
    run_id_save_list = []
    for task_name, (exp_name, run_id_list) in tasks_maps.items():
        for run_id in run_id_list:
            task_name_list.append(task_name)
            exp_name_list.append(exp_name)
            run_id_save_list.append(run_id)
    model_df = pd.DataFrame.from_dict({'task': task_name_list, 'exp_name': exp_name_list, 'run_id': run_id_save_list})

    matrix_suffix = '240928'

    dataset_name = 'tdw_1m_20240206_val_0_04'
    record_layers = ['layer1.1.relu', 'layer2.1.relu', 'layer3.1.relu', 'layer4.1.relu', 'avgpool']
    for layer in record_layers:
        print(f'Processing layer {layer} ...')
        layer_name = layer.replace('.', '_')

        for idx2 in range(len(model_df)):
            exp_name1, run_id1 = model_df.iloc[idx1]['exp_name'], model_df.iloc[idx1]['run_id']
            exp_name2, run_id2 = model_df.iloc[idx2]['exp_name'], model_df.iloc[idx2]['run_id']
            model_id1, model_id2 = f'{exp_name1}_{run_id1:04d}', f'{exp_name2}_{run_id2:04d}'
            
            score_path = Path(DATA_DIR).joinpath('rsa', f'cka_{model_id1}_{model_id2}_{layer_name}_{dataset_name}.npy')
            if score_path.is_file():
                CKA_score = np.load(score_path)
                print(f'load score successfully: {CKA_score}')
            else:
                act1 = np.load(Path(EXP_DIR).joinpath(exp_name1, f'run_{run_id1:04d}', f'act_{layer_name}_{dataset_name}.npy'))
                act2 = np.load(Path(EXP_DIR).joinpath(exp_name2, f'run_{run_id2:04d}', f'act_{layer_name}_{dataset_name}.npy'))
                CKA_score = linear_CKA(act1, act2)
                np.save(score_path, CKA_score)
                print(f'save successfully: {CKA_score}')
