import os
import copy
from config_global import EXP_DIR


def setup_config_list():
    base_config = {
        'seed': 0,
        'run_id': 0,
        'batch_size': 64,
        'lr': 1e-3,
        'max_batch': 1000,
        'eval_per': 10,
        'group_name': 'multi_task',
        'tasks': [
            'category_class',
            'object_class',
            'rotation_reg',
            'size_reg',
            'translation_reg',
            ],
        'model_archi': 'resnet18',
        'experiment_name': 'multi_task_0610',
        'save_path': './experiments/',
        }

    task_set_dict = {
        'multi_task': ['category_class', 'object_class', 'rotation_reg', 'size_reg', 'translation_reg'],
        'categorization': ['category_class'],
        'multi_task_wo_object_class': ['category_class', 'rotation_reg', 'size_reg', 'translation_reg'],
        'size_reg': ['size_reg'],
        'translation_reg': ['translation_reg'],
        'rotation_reg': ['rotation_reg'],
    }
    seed_list = [0, 1, 2, 3, 4]
    
    # setting up config list
    config_list = []
    run_id = 0
    for group_n, task_set in task_set_dict.items():
        for seed in seed_list:
            cfg = copy.deepcopy(base_config)
            cfg['group_name'] = group_n
            cfg['tasks'] = task_set
            cfg['seed'] = seed

            cfg['save_path'] = os.path.join(EXP_DIR, cfg['experiment_name'], f'run_{run_id:04d}')
            cfg['run_id'] = run_id
            config_list.append(cfg)
            run_id += 1
    
    return config_list