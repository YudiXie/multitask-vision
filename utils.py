import os

import yaml

def save_config(config, save_folder='./exp_configs'):
    """
    save a config dictrionary to a yaml file in the folder
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    yaml_file_path = os.path.join(save_folder, 'config.yml')
    with open(yaml_file_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    return yaml_file_path


def load_config(yaml_file_path='config.yml'):
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config