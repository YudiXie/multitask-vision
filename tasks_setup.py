import torch
import torch.nn as nn


def two_units_sin_cos_mse(output, target):
    """
    parameters:
        output: torch.Tensor, shape(batch_size, 6), sin and cos of 3 rotation angles
        target: torch.Tensor, shape(batch_size, 3), 3 rotation angles in degrees
    return:
        loss: torch.Tensor, shape(1), mean squared error between output and target
    """
    target_rad = torch.deg2rad(target)
    target_sin = torch.sin(target_rad)
    target_cos = torch.cos(target_rad)
    target_all = torch.cat([target_sin, target_cos], dim=1)
    return torch.mean((output - target_all)**2)

# Task relevant constants
cat_reduced_tasks = ['cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8']

# mapping from individual task names to the corresponding target names in the sample (or dataframe column names)
task2targets_name = {
    'cat2': ['cat_label_reduce2'],
    'cat3': ['cat_label_reduce3'],
    'cat4': ['cat_label_reduce4'],
    'cat5': ['cat_label_reduce5'],
    'cat6': ['cat_label_reduce6'],
    'cat7': ['cat_label_reduce7'],
    'cat8': ['cat_label_reduce8'],
    'category_class': ['category_label'],
    'object_class': ['object_label'],
    'rotation_reg': ['rxy_semantic', 'rxz_semantic', 'ryz_semantic'],
    'rotation_reg_tdw': ['euler_1_proc', 'euler_2_proc', 'euler_3_proc'], # for TDW dataset
    'distance_reg': ['rel_pos_z'], # for TDW dataset
    'size_reg': ['s'],
    'translation_reg': ['rel_pos_x', 'rel_pos_y'],
    'rotation_reg_tdw_two_units_sin_cos_mse': ['rel_rot_euler_0', 'rel_rot_euler_1', 'rel_rot_euler_2'],
}

# mapping from individual task names to the corresponding loss functions to be used
task2loss_func = {
    'cat2': nn.CrossEntropyLoss(),
    'cat3': nn.CrossEntropyLoss(),
    'cat4': nn.CrossEntropyLoss(),
    'cat5': nn.CrossEntropyLoss(),
    'cat6': nn.CrossEntropyLoss(),
    'cat7': nn.CrossEntropyLoss(),
    'cat8': nn.CrossEntropyLoss(),
    'category_class': nn.CrossEntropyLoss(),
    'object_class': nn.CrossEntropyLoss(),
    'rotation_reg': nn.MSELoss(),
    'rotation_reg_tdw': nn.MSELoss(),
    'distance_reg': nn.MSELoss(),
    'size_reg': nn.MSELoss(),
    'translation_reg': nn.MSELoss(),
    'rotation_reg_tdw_two_units_sin_cos_mse': two_units_sin_cos_mse,
}

# mapping from individual task names to the output units in the model
# for small TDW dataset and HvM dataset
task2output_range_small = {
    'cat2': (0, 2),
    'cat3': (0, 3),
    'cat4': (0, 4),
    'cat5': (0, 5),
    'cat6': (0, 6),
    'cat7': (0, 7),
    'cat8': (0, 8), # equvalent to category_class
    'category_class': (0, 8),
    'object_class': (8, 72),
    'rotation_reg': [72, 75],
    'rotation_reg_tdw': [72, 75],
    'size_reg': [75, 76],
    'distance_reg': [75, 76],
    'translation_reg': [76, 78],
    'rotation_reg_tdw_two_units_sin_cos_mse': [78, 84],
}

# for large TDW dataset
task2output_range_large = {
    'category_class': (0, 117),
    'object_class': (117, 704),
    'rotation_reg': [704, 707],
    'rotation_reg_tdw': [704, 707],
    'size_reg': [707, 708],
    'distance_reg': [707, 708],
    'translation_reg': [708, 710],
    'rotation_reg_tdw_two_units_sin_cos_mse': [710, 716],
}

task2output_range_large_new = {
    'category_class': [0, 117],
    'object_class': [117, 665],
    'distance_reg': [665, 666],
    'translation_reg': [666, 668],
    'rotation_reg_tdw_two_units_sin_cos_mse': [668, 674],
}


def get_output_info(dataset_name):
    """
    determine the output dimention and the output range based on the dataset
    args:
        dataset_name: str, name of the dataset
    """

    if dataset_name in ['TDW_large20230907', 'TDW_large20240112']:
        # TDW large dataset
        output_number = 716 # 117 + 587 + 3 + 1 + 2 + 6
        task2output_range = task2output_range_large
    elif dataset_name == 'TDW' or dataset_name == 'HvM':
        # TDW small dataset and HvM dataset
        output_number = 84  # 8 + 64 + 3 + 1 + 2 + 6
        task2output_range = task2output_range_small
    elif dataset_name in ['tdw_1m_20240206', 
                          'tdw_10m_20240208', 
                          'tdw_100m_20240222',
                          'tdw_1m_1c_n03001627_20240711',
                          'tdw_1m_obj_centered_20240812',
                          ]:
        output_number = 674  # 117 + 548 + 1 + 2 + 6
        task2output_range = task2output_range_large_new
    else:
        raise NotImplementedError(f'Unknown dataset: {dataset_name}')
    
    return output_number, task2output_range
