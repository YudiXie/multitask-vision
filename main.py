import copy

from train import train_model

if __name__ == '__main__':

    config={
        "batch_size": 32,
        "lr": 1e-3,
        "max_batch": 500,
        "eval_per": 10,
        "tasks": [
            'category_class',
            'object_class',
            'rotation_reg',
            'size_reg',
            'translation_reg',
            ],
        }
    
    lr_list = [0.4 * 1e-3, 0.7 * 1e-3, 1e-3, 1.3 * 1e-3, 1.6 * 1e-3]
    for lr in lr_list:
        train_config = copy.deepcopy(config)
        train_config['lr'] = lr
        train_model(train_config)
