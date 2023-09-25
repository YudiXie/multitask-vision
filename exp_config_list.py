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
    'restart_from_checkpoint': True,
    'checkpoint_per': 100,
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
    'train_dataset_fraction': 1.0,
    }


def random_models0630():
    # only used to score the random models
    exp_config = copy.deepcopy(base_config)
    exp_config['experiment_name'] = 'random_models0630'
    exp_config['dataset_name'] = 'HvM'

    seed_list = [0, 1, 2, 3, 4]
    
    # setting up config list
    config_list = []
    run_id = 0
    for seed in seed_list:
        cfg = copy.deepcopy(exp_config)
        cfg['seed'] = seed

        cfg['save_path'] = os.path.join(EXP_DIR, cfg['experiment_name'], f'run_{run_id:04d}')
        cfg['run_id'] = run_id
        config_list.append(cfg)
        run_id += 1
    return config_list


def multi_task_0620():
    # compare models with different training targets
    exp_config = copy.deepcopy(base_config)
    exp_config['experiment_name'] = 'multi_task_0620'
    exp_config['dataset_name'] = 'HvM'

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


def multi_task_nopret_longtrain_0629():
    config_list = multi_task_0620()
    for config in config_list:
        config['experiment_name'] = 'multi_task_nopret_longtrain_0629'
        config['pretrain_init'] = False
        config['max_batch'] = 5000
        run_id = config['run_id']
        config['save_path'] = os.path.join(EXP_DIR, config['experiment_name'], f'run_{run_id:04d}')
    return config_list


def cat_diff_0623():
    # compare models trained with categorization tasks with different number of output units
    exp_config = copy.deepcopy(base_config)
    exp_config['experiment_name'] = 'cat_diff_0623'
    exp_config['dataset_name'] = 'HvM'

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


def cat_diff_nopret_longtrain_0629():
    config_list = cat_diff_0623()
    for config in config_list:
        config['experiment_name'] = 'cat_diff_nopret_longtrain_0629'
        config['pretrain_init'] = False
        config['max_batch'] = 5000
        run_id = config['run_id']
        config['save_path'] = os.path.join(EXP_DIR, config['experiment_name'], f'run_{run_id:04d}')
    return config_list


def multi_task_tdw_0817():
    # compare models with different training targets
    exp_config = copy.deepcopy(base_config)
    exp_config['experiment_name'] = 'multi_task_tdw_0817'
    exp_config['dataset_name'] = 'TDW'

    task_set_dict = {
        'distance_reg': ['distance_reg'],
        'translation_reg': ['translation_reg'],
        'rotation_reg': ['rotation_reg_tdw'],

        'distance_translation': ['distance_reg', 'translation_reg'],
        'distance_rotation': ['distance_reg', 'rotation_reg_tdw'],
        'translation_rotation': ['translation_reg', 'rotation_reg_tdw'],

        'distance_translation_rotation': ['distance_reg', 'translation_reg', 'rotation_reg_tdw'],

        'categorization': ['category_class'],
        'multi_task': ['category_class', 'rotation_reg_tdw', 'distance_reg', 'translation_reg'],
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


def cat_diff_tdw_0820():
    # compare models trained with categorization tasks with different number of output units
    exp_config = copy.deepcopy(base_config)
    exp_config['experiment_name'] = 'cat_diff_tdw_0820'
    exp_config['dataset_name'] = 'TDW'

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


def multi_task_tdw_multiscene_0826():
    config_list = multi_task_tdw_0817()
    for config in config_list:
        config['experiment_name'] = 'multi_task_tdw_multiscene_0826'
        run_id = config['run_id']
        config['save_path'] = os.path.join(EXP_DIR, config['experiment_name'], f'run_{run_id:04d}')
    return config_list


def cat_diff_tdw_multiscene_0826():
    config_list = cat_diff_tdw_0820()
    for config in config_list:
        config['experiment_name'] = 'cat_diff_tdw_multiscene_0826'
        run_id = config['run_id']
        config['save_path'] = os.path.join(EXP_DIR, config['experiment_name'], f'run_{run_id:04d}')
    return config_list


def multi_task_tdw_multiscene_hdri_0906():
    config_list = multi_task_tdw_0817()
    for config in config_list:
        config['experiment_name'] = 'multi_task_tdw_multiscene_hdri_0906'
        run_id = config['run_id']
        config['save_path'] = os.path.join(EXP_DIR, config['experiment_name'], f'run_{run_id:04d}')
    return config_list


def cat_diff_tdw_multiscene_hdri_0906():
    config_list = cat_diff_tdw_0820()
    for config in config_list:
        config['experiment_name'] = 'cat_diff_tdw_multiscene_hdri_0906'
        run_id = config['run_id']
        config['save_path'] = os.path.join(EXP_DIR, config['experiment_name'], f'run_{run_id:04d}')
    return config_list


def multi_task_tdw_large20230907_0919():
    config_list = multi_task_tdw_0817()
    new_config_list = []
    for config in config_list:
        if config['seed'] >= 3:
            continue
        
        config['experiment_name'] = 'multi_task_tdw_large20230907_0919'
        run_id = config['run_id']
        config['save_path'] = os.path.join(EXP_DIR, config['experiment_name'], f'run_{run_id:04d}')

        config['dataset_name'] = 'TDW_large20230907'
        config['max_batch'] = 200000  # run thorugh the dataset ~10 times with batchsize 64
        config['eval_per'] = 1000
        config['checkpoint_per'] = 1000
        
        new_config_list.append(config)
    return new_config_list


def multi_task_tdw_large20230907_nopret_0925():
    # compare models with different training targets
    exp_config = copy.deepcopy(base_config)
    exp_config['experiment_name'] = 'multi_task_tdw_large20230907_nopret_0925'
    exp_config['dataset_name'] = 'TDW_large20230907'
    exp_config['max_batch'] = 200000  # run thorugh the dataset ~10 times with batchsize 64
    exp_config['eval_per'] = 1000
    exp_config['checkpoint_per'] = 1000
    exp_config['pretrain_init'] = False

    task_set_dict = {
        'distance_reg': ['distance_reg'],
        'translation_reg': ['translation_reg'],
        'rotation_reg': ['rotation_reg_tdw'],

        'distance_translation': ['distance_reg', 'translation_reg'],
        'distance_rotation': ['distance_reg', 'rotation_reg_tdw'],
        'translation_rotation': ['translation_reg', 'rotation_reg_tdw'],

        'distance_translation_rotation': ['distance_reg', 'translation_reg', 'rotation_reg_tdw'],

        'category_class': ['category_class'],
        'object_class': ['object_class'],
        'cat_obj_class_all_latents': ['category_class', 'object_class', 'rotation_reg_tdw', 'distance_reg', 'translation_reg'],
    }
    seed_list = [0, 1, 2]
    
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


def multi_task_tdw_large20230907_nopret_dis_scaling_0925():
    # compare models with different training targets
    exp_config = copy.deepcopy(base_config)
    exp_config['experiment_name'] = 'multi_task_tdw_large20230907_nopret_dis_scaling_0925'
    exp_config['dataset_name'] = 'TDW_large20230907'
    exp_config['max_batch'] = 200000  # run thorugh the dataset ~10 times with batchsize 64
    exp_config['eval_per'] = 1000
    exp_config['checkpoint_per'] = 1000
    exp_config['pretrain_init'] = False
    exp_config['tasks'] = ['distance_reg']
    seed_list = [0, 1, 2]
    
    fractions = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]

    # setting up config list
    config_list = []
    run_id = 0
    for frac in fractions:
        for seed in seed_list:
            cfg = copy.deepcopy(exp_config)
            cfg['group_name'] = f'frac_{frac}'
            cfg['train_dataset_fraction'] = frac
            cfg['seed'] = seed

            cfg['save_path'] = os.path.join(EXP_DIR, cfg['experiment_name'], f'run_{run_id:04d}')
            cfg['run_id'] = run_id
            config_list.append(cfg)
            run_id += 1
    return config_list