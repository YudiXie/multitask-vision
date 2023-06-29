import os
import copy
from config_global import EXP_DIR


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
    # 'experiment_name': 'multi_task_0620',
    'save_path': './experiments/',
    'pretrain_init': True,
    }


def multi_task_0620():
    # compare models with different training targets
    exp_config = copy.deepcopy(base_config)
    exp_config['experiment_name'] = 'multi_task_0620'

    task_set_dict = {
        'size_reg': ['size_reg'],
        'translation_reg': ['translation_reg'],
        'rotation_reg': ['rotation_reg'],

        'size_translation': ['size_reg', 'translation_reg'],
        'size_rotation': ['size_reg', 'rotation_reg'],
        'translation_rotation': ['translation_reg', 'rotation_reg'],

        'size_translation_rotation': ['size_reg', 'translation_reg', 'rotation_reg'],

        'categorization': ['category_class'],
        'multi_task_wo_object_class': ['category_class', 'rotation_reg', 'size_reg', 'translation_reg'],
        'multi_task': ['category_class', 'object_class', 'rotation_reg', 'size_reg', 'translation_reg'],
    }
    seed_list = [0, 1, 2, 3, 4]
    
    # setting up config list
    config_list = []
    run_id = 0
    for group_n, task_set in task_set_dict.items():
        for seed in seed_list:
            cfg = copy.deepcopy(exp_config)
            cfg['group_name'] = group_n
            cfg['tasks'] = task_set
            cfg['seed'] = seed

            cfg['save_path'] = os.path.join(EXP_DIR, cfg['experiment_name'], f'run_{run_id:04d}')
            cfg['run_id'] = run_id
            config_list.append(cfg)
            run_id += 1
    return config_list


def multi_task_nopret_0629():
    config_list = multi_task_0620()
    for config in config_list:
        config['experiment_name'] = 'multi_task_nopret_0629'
        config['pretrain_init'] = False
        run_id = config['run_id']
        config['save_path'] = os.path.join(EXP_DIR, config['experiment_name'], f'run_{run_id:04d}')
    return config_list


def cat_diff_0623():
    # compare models trained with categorization tasks with different number of output units
    exp_config = copy.deepcopy(base_config)
    exp_config['experiment_name'] = 'cat_diff_0623'

    task_set_dict = {
        'cat2': ['cat2'],
        'cat3': ['cat3'],
        'cat4': ['cat4'],
        'cat5': ['cat5'],
        'cat6': ['cat6'],
        'cat7': ['cat7'],
        'cat8': ['cat8'],
    }
    seed_list = [0, 1, 2, 3, 4]
    
    # setting up config list
    config_list = []
    run_id = 0
    for group_n, task_set in task_set_dict.items():
        for seed in seed_list:
            cfg = copy.deepcopy(exp_config)
            cfg['group_name'] = group_n
            cfg['tasks'] = task_set
            cfg['seed'] = seed

            cfg['save_path'] = os.path.join(EXP_DIR, cfg['experiment_name'], f'run_{run_id:04d}')
            cfg['run_id'] = run_id
            config_list.append(cfg)
            run_id += 1
    return config_list


def cat_diff_nopret_0629():
    config_list = cat_diff_0623()
    for config in config_list:
        config['experiment_name'] = 'cat_diff_nopret_0629'
        config['pretrain_init'] = False
        run_id = config['run_id']
        config['save_path'] = os.path.join(EXP_DIR, config['experiment_name'], f'run_{run_id:04d}')
    return config_list
