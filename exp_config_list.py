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
    'use_amp': False,
    'save_inter_model': [],
    'score_model_nums': [],
    }


def change_config(config_list_func, new_exp_name, change_kwargs):
    """
    set attributes in change_kwargs to all config in config_list,
    and generate a new list of experimental configs, with different epxeriment names and save paths
    params:
        config_list_func: a function that returns a list of config, can be empty
        new_exp_name: str, new experiment name
        change_kwargs: dict, new attributes to be added to all config
    return:
        list of new configs
    """
    config_list = config_list_func()
    for cfg in config_list:
        for k, v in change_kwargs.items():
            assert k not in ['experiment_name', 'run_id', 'save_path'], f'key {k} cannot be changed'
            assert k in cfg, f'key {k} not in config'
            cfg[k] = v
        
        # mandatory changes
        cfg['experiment_name'] = new_exp_name
        run_id = cfg['run_id']
        cfg['save_path'] = os.path.join(EXP_DIR, cfg['experiment_name'], f'run_{run_id:04d}')

    return config_list


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


def multi_task_tdw_1m20240206_nopret_0214():
    # compare models with different training targets
    exp_config = copy.deepcopy(base_config)
    exp_config['experiment_name'] = 'multi_task_tdw_1m20240206_nopret_0214'
    exp_config['dataset_name'] = 'tdw_1m_20240206'
    exp_config['max_batch'] = 200000  # run thorugh the dataset ~10 times with batchsize 64
    exp_config['eval_per'] = 5000
    exp_config['checkpoint_per'] = 1000
    exp_config['pretrain_init'] = False

    task_set_dict = {
        'distance_reg': ['distance_reg'],
        'translation_reg': ['translation_reg'],
        'rotation_reg': ['rotation_reg_tdw_two_units_sin_cos_mse'],

        'distance_translation': ['distance_reg', 'translation_reg'],
        'distance_rotation': ['distance_reg', 'rotation_reg_tdw_two_units_sin_cos_mse'],
        'translation_rotation': ['translation_reg', 'rotation_reg_tdw_two_units_sin_cos_mse'],

        'distance_translation_rotation': ['distance_reg', 'translation_reg', 'rotation_reg_tdw_two_units_sin_cos_mse'],

        'category_class': ['category_class'],
        'object_class': ['object_class'],
        'cat_obj_class_all_latents': ['category_class', 'object_class', 'rotation_reg_tdw_two_units_sin_cos_mse', 'distance_reg', 'translation_reg'],
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


def multi_task_tdw_1m20240206_resnet50_nopret_0219():
    change_kwargs = {
        'model_archi': 'resnet50',
    }
    return change_config(multi_task_tdw_1m20240206_nopret_0214, 
                         'multi_task_tdw_1m20240206_resnet50_nopret_0219', 
                         change_kwargs)


def multi_task_tdw_10m20240208_resnet50_nopret_0221():
    change_kwargs = {
        'dataset_name': 'tdw_10m_20240208',
        'max_batch': 500000,  # run thorugh the 10m dataset ~3 times
        'eval_per': 20000,
    }
    return change_config(multi_task_tdw_1m20240206_resnet50_nopret_0219, 
                         'multi_task_tdw_10m20240208_resnet50_nopret_0221', 
                         change_kwargs)


def multi_task_tdw_10m20240208_resnet50_nopret_1mb_0223():
    change_kwargs = {
        'max_batch': 1000000,  # run thorugh the 10m dataset ~6 times
    }
    return change_config(multi_task_tdw_10m20240208_resnet50_nopret_0221, 
                         'multi_task_tdw_10m20240208_resnet50_nopret_1mb_0223', 
                         change_kwargs)


def pretrain_and_random_resnet50_0220():
    # only used to score the random models
    exp_config = copy.deepcopy(base_config)
    exp_config['experiment_name'] = 'pretrain_and_random_resnet50_0220'
    exp_config['dataset_name'] = 'tdw_1m_20240206'
    exp_config['model_archi'] = 'resnet50'

    group_list = ['random', 'imagenet1k_pretrain']
    seed_list = [0, 1, 2, 3, 4]
    
    # setting up config list
    config_list = []
    run_id = 0
    for group_n in group_list:
        for seed in seed_list:
            cfg = copy.deepcopy(exp_config)
            cfg['group_name'] = group_n
            cfg['seed'] = seed

            cfg['save_path'] = os.path.join(EXP_DIR, cfg['experiment_name'], f'run_{run_id:04d}')
            cfg['run_id'] = run_id
            config_list.append(cfg)
            run_id += 1
    return config_list


def pretrain_and_random_resnet18_0220():
    change_kwargs = {
        'model_archi': 'resnet18',
    }
    return change_config(pretrain_and_random_resnet50_0220, 
                         'pretrain_and_random_resnet18_0220', 
                         change_kwargs)


def dis_scaling_tdw_10m20240208_resnet50_nopret_1mb_0306():
    change_kwargs = {
        'model_archi': 'resnet50',
        'dataset_name': 'tdw_10m_20240208',
        'max_batch': 1000000,  # run thorugh the 10m dataset ~6 times
        'eval_per': 20000,
    }
    return change_config(multi_task_tdw_large20230907_nopret_dis_scaling_0925, 
                         'dis_scaling_tdw_10m20240208_resnet50_nopret_1mb_0306', 
                         change_kwargs)


def allcat_alllat_tdw_10m20240208_resnet50_nopret_1mb_0306():
    change_kwargs = {
        'tasks': ['category_class', 'object_class', 'rotation_reg_tdw_two_units_sin_cos_mse', 'distance_reg', 'translation_reg'],
    }
    return change_config(dis_scaling_tdw_10m20240208_resnet50_nopret_1mb_0306, 
                         'allcat_alllat_tdw_10m20240208_resnet50_nopret_1mb_0306', 
                         change_kwargs)


def dis_scaling_tdw_10m20240208_resnet50_nopret_500kb_0710():
    # just to test the new pytorch version
    change_kwargs = {
        'max_batch': 500000,
        'use_amp': True,
    }
    return change_config(dis_scaling_tdw_10m20240208_resnet50_nopret_1mb_0306, 
                         'dis_scaling_tdw_10m20240208_resnet50_nopret_500kb_0710', 
                         change_kwargs)


def acal_tdw_nopret_dis_scaling_240711():
    exp_config = copy.deepcopy(base_config)
    exp_config['experiment_name'] = 'acal_tdw_nopret_dis_scaling_240711'
    exp_config['model_archi'] = 'resnet50'
    exp_config['dataset_name'] = 'tdw_100m_20240222'
    exp_config['max_batch'] = 3000000  # run thorugh the dataset ~2 times with batchsize 64, ~ 30h to run
    exp_config['eval_per'] = 30000
    exp_config['checkpoint_per'] = 5000
    exp_config['pretrain_init'] = False
    exp_config['tasks'] = ['category_class', 'object_class', 'rotation_reg_tdw_two_units_sin_cos_mse', 'distance_reg', 'translation_reg']
    
    seed_list = [0, 1, 2]
    fractions = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]

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


def ctrl_var_target_dist_240712():
    exp_config = copy.deepcopy(base_config)
    exp_config['experiment_name'] = 'ctrl_var_target_dist_240712'

    exp_config['max_batch'] = 200000  # run thorugh the dataset ~10 times with batchsize 64
    exp_config['eval_per'] = 5000
    exp_config['checkpoint_per'] = 1000
    exp_config['pretrain_init'] = False
    
    dset_list = ['tdw_1m_20240206', 
                 'tdw_1m_1c_n03001627_20240711']
    task_set_dict = {
        'distance_reg': ['distance_reg'],
        'translation_reg': ['translation_reg'],
        'rotation_reg': ['rotation_reg_tdw_two_units_sin_cos_mse'],
        'distance_translation_rotation': ['distance_reg', 'translation_reg', 'rotation_reg_tdw_two_units_sin_cos_mse'],
    }
    seed_list = [0, 1, 2]

    # setting up config list
    config_list = []
    run_id = 0
    for dset in dset_list:
        for task_set_n, task_set in task_set_dict.items():
            for seed in seed_list:
                cfg = copy.deepcopy(exp_config)
                cfg['dataset_name'] = dset
                cfg['task_set_name'] = task_set_n
                cfg['tasks'] = task_set
                cfg['seed'] = seed

                cfg['save_path'] = os.path.join(EXP_DIR, cfg['experiment_name'], f'run_{run_id:04d}')
                cfg['run_id'] = run_id
                config_list.append(cfg)
                run_id += 1
    return config_list


def multi_task_tdw_1m20240206_0718():
    # compare models with different training targets
    exp_config = copy.deepcopy(base_config)
    exp_config['experiment_name'] = 'multi_task_tdw_1m20240206_0718'
    exp_config['dataset_name'] = 'tdw_1m_20240206'
    exp_config['max_batch'] = 500000  # run thorugh the dataset ~30 times with batchsize 64
    exp_config['eval_per'] = 10000
    exp_config['checkpoint_per'] = 1000
    exp_config['pretrain_init'] = False
    exp_config['save_inter_model'] = [100000, 200000, 500000, 1000000, 1500000]

    task_set_dict = {
        'distance_reg': ['distance_reg'],
        'translation_reg': ['translation_reg'],
        'rotation_reg': ['rotation_reg_tdw_two_units_sin_cos_mse'],

        'distance_translation': ['distance_reg', 'translation_reg'],
        'distance_rotation': ['distance_reg', 'rotation_reg_tdw_two_units_sin_cos_mse'],
        'translation_rotation': ['translation_reg', 'rotation_reg_tdw_two_units_sin_cos_mse'],

        'distance_translation_rotation': ['distance_reg', 'translation_reg', 'rotation_reg_tdw_two_units_sin_cos_mse'],

        'category_class': ['category_class'],
        'object_class': ['object_class'],
        'cat_obj_class_all_latents': ['category_class', 'object_class', 'rotation_reg_tdw_two_units_sin_cos_mse', 'distance_reg', 'translation_reg'],
    }
    seed_list = [0, 1, 2, 3, 4, 5, 6, 7]
    
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


def multi_task_tdw_1m20240206_moresavings_0801():
    change_kwargs = {
        'save_inter_model': [20000, 40000, 60000, 80000, 100000, 150000, 200000, 500000],
    }
    return change_config(multi_task_tdw_1m20240206_0718, 
                         'multi_task_tdw_1m20240206_moresavings_0801', 
                         change_kwargs)


def multi_task_resnet50_tdw_10m20240208_0802():
    # compare models with different training targets
    exp_config = copy.deepcopy(base_config)
    exp_config['experiment_name'] = 'multi_task_resnet50_tdw_10m20240208_0802'
    exp_config['dataset_name'] = 'tdw_10m_20240208'
    exp_config['max_batch'] = 1000000
    exp_config['eval_per'] = 20000
    exp_config['checkpoint_per'] = 1000
    exp_config['pretrain_init'] = False
    exp_config['model_archi'] = 'resnet50'
    exp_config['save_inter_model'] = [20000, 40000, 60000, 80000, 100000, 150000, 200000, 500000, 700000, 1000000]
    exp_config['score_model_nums'] = [20000, 40000, 60000, 80000, 100000, 150000, 200000, 500000, 700000, 1000000]
    exp_config['train_dataset_fraction'] = 0.3

    task_set_dict = {
        'distance_reg': ['distance_reg'],
        'translation_reg': ['translation_reg'],
        'rotation_reg': ['rotation_reg_tdw_two_units_sin_cos_mse'],

        'distance_translation': ['distance_reg', 'translation_reg'],
        'distance_rotation': ['distance_reg', 'rotation_reg_tdw_two_units_sin_cos_mse'],
        'translation_rotation': ['translation_reg', 'rotation_reg_tdw_two_units_sin_cos_mse'],

        'distance_translation_rotation': ['distance_reg', 'translation_reg', 'rotation_reg_tdw_two_units_sin_cos_mse'],

        'category_class': ['category_class'],
        'object_class': ['object_class'],
        'cat_obj_class_all_latents': ['category_class', 'object_class', 'rotation_reg_tdw_two_units_sin_cos_mse', 'distance_reg', 'translation_reg'],
    }
    seed_list = [0, 1, 2, 3, 4, 5, 6, 7]
    
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


def multi_task_resnet50_tdw_10m20240208_earlier_0817():
    change_kwargs = {
        'max_batch': 10000,
        'eval_per': 2000,
        'checkpoint_per': 200,
        'save_inter_model': [2000, 4000, 10000],
        'score_model_nums': [2000, 4000, 10000],
    }
    return change_config(multi_task_resnet50_tdw_10m20240208_0802, 
                         'multi_task_resnet50_tdw_10m20240208_earlier_0817', 
                         change_kwargs)


# compare with a dataset that has little variations in translation
def ctrl_trans_var_240814():
    exp_config = copy.deepcopy(base_config)
    exp_config['experiment_name'] = 'ctrl_trans_var_240814'

    exp_config['max_batch'] = 200000  # run thorugh the dataset ~10 times with batchsize 64
    exp_config['eval_per'] = 5000
    exp_config['checkpoint_per'] = 1000
    exp_config['pretrain_init'] = False
    
    dset_list = ['tdw_1m_20240206', 
                 'tdw_1m_obj_centered_20240812']
    task_set_dict = {
        'distance_reg': ['distance_reg'],
        'rotation_reg': ['rotation_reg_tdw_two_units_sin_cos_mse'], # rotation is affected in tdw_1m_obj_centered_20240812 but still adding it here
        'category_class': ['category_class'],
        'object_class': ['object_class'],
    }
    seed_list = [0, 1, 2, 3, 4]

    # setting up config list
    config_list = []
    run_id = 0
    for dset in dset_list:
        for task_set_n, task_set in task_set_dict.items():
            for seed in seed_list:
                cfg = copy.deepcopy(exp_config)
                cfg['dataset_name'] = dset
                cfg['task_set_name'] = task_set_n
                cfg['tasks'] = task_set
                cfg['seed'] = seed

                cfg['save_path'] = os.path.join(EXP_DIR, cfg['experiment_name'], f'run_{run_id:04d}')
                cfg['run_id'] = run_id
                config_list.append(cfg)
                run_id += 1
    return config_list


def resnet50_tdw100m_scaling_240822():
    exp_config = copy.deepcopy(base_config)
    exp_config['experiment_name'] = 'resnet50_tdw100m_scaling_240822'
    exp_config['model_archi'] = 'resnet50'
    exp_config['dataset_name'] = 'tdw_100m_20240222'
    exp_config['max_batch'] = 1500000
    exp_config['eval_per'] = 30000
    exp_config['checkpoint_per'] = 5000
    exp_config['pretrain_init'] = False
    exp_config['tasks'] = ['category_class', 'object_class', 'rotation_reg_tdw_two_units_sin_cos_mse', 'distance_reg', 'translation_reg']
    exp_config['save_inter_model'] = [100000, 200000, 500000, 1000000, 1500000]
    
    fractions = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]
    task_set_dict = {
        'distance_reg': ['distance_reg'],
        'translation_reg': ['translation_reg'],
        'rotation_reg': ['rotation_reg_tdw_two_units_sin_cos_mse'],
        'category_class': ['category_class'],
        'cat_obj_class_all_latents': ['category_class', 'object_class', 'rotation_reg_tdw_two_units_sin_cos_mse', 'distance_reg', 'translation_reg'],
    }
    seed_list = [0, 1, 2]

    # setting up config list
    config_list = []
    run_id = 0
    for frac in fractions:
        for task_set_n, task_set in task_set_dict.items():
            for seed in seed_list:
                cfg = copy.deepcopy(exp_config)
                cfg['train_dataset_fraction'] = frac
                cfg['task_set_name'] = task_set_n
                cfg['tasks'] = task_set
                cfg['seed'] = seed

                cfg['save_path'] = os.path.join(EXP_DIR, cfg['experiment_name'], f'run_{run_id:04d}')
                cfg['run_id'] = run_id
                config_list.append(cfg)
                run_id += 1
    return config_list


def imagenet1k_0902():
    # copy settings from multi_task_tdw_1m20240206_0718
    exp_config = copy.deepcopy(base_config)
    exp_config['experiment_name'] = 'imagenet1k_0902'
    exp_config['dataset_name'] = 'ImageNet1K'
    exp_config['max_batch'] = 500000
    exp_config['eval_per'] = 10000
    exp_config['checkpoint_per'] = 1000
    exp_config['pretrain_init'] = False
    exp_config['tasks'] = ['category_class', ]
    
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
