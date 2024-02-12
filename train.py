import os
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import wandb

from config_global import DEVICE, NP_SEED, TCH_SEED
from dataset import HVMDataset, TDWDataset
from utils import load_config, log_complete
from tasks_setup import cat_reduced_tasks, task2loss_func, task2targets_name, get_output_info


# used to set the weights of the different tasks by hand
# note that when less than 5 tasks are used, the weights should be adjusted
# so that they sum to 1
# task2weights = {
#     'category_class': 0.2,
#     'object_class': 0.2,
#     'rotation_reg': 0.2,
#     'size_reg': 0.2,
#     'translation_reg': 0.2,
# }

# Data preprocessing
IMN_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def get_dataloader(dataset_name, is_train, batch_size, transform, dataset_fraction=1.0):
    "Get a training dataloader"    
    if is_train:
        split = 'train'
    else:
        split = 'val'

    if dataset_name == 'HvM':
        assert dataset_fraction == 1.0, 'HvM dataset does not support dataset_fraction'
        dataset = HVMDataset(split=split, transform=transform)
    elif dataset_name == 'TDW':
        dataset = TDWDataset(split=split, transform=transform,
                             fraction=dataset_fraction,)
    elif dataset_name == 'TDW_large20230907':
        dataset = TDWDataset(root_dir='./data/tdw_image_dataset_large_20230907', 
                             split=split,
                             transform=transform,
                             fraction=dataset_fraction,)
    elif dataset_name == 'TDW_large20240112':
        dataset = TDWDataset(root_dir='/om2/user/yu_xie/data/tdw_images/tdw_image_dataset_large', 
                             split=split,
                             transform=transform,
                             fraction=dataset_fraction,)
    elif dataset_name == 'tdw_1m_20240206':
        dataset = TDWDataset(root_dir='/om/user/yu_xie/data/tdw_images/tdw_image_dataset_1m_20240206', 
                             split=split,
                             transform=transform,
                             fraction=dataset_fraction,)
    else:
        raise NotImplementedError(f'Unknown dataset: {dataset_name}')
    
    loader = torch.utils.data.DataLoader(dataset=dataset, 
                                         batch_size=batch_size, 
                                         shuffle=True if is_train else False, 
                                         pin_memory=True, 
                                         num_workers=8,
                                         drop_last=True,
                                         )
    return loader


def validate_model(model,
                   valid_dl,
                   task_list,
                   task2output_range,
                   log_images=False,
                   batch_idx=0):
    "Compute performance of the model on the validation dataset and log a wandb.Table"
    val_loss = 0.0
    val_task_loss = {task: 0.0 for task in task_list}
    category_correct = 0
    object_correct = 0
    cat_red_correct = 0  # assumes that only one reduece categorization task is used
    image_ct = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(valid_dl):
            inputs = data['image'].to(DEVICE)
            outputs = model(inputs)
            batch_size = len(inputs)
            image_ct += batch_size

            batch_loss_dict = {}
            for task in task_list:
                task_targets = []
                for target_name in task2targets_name[task]:
                    task_targets.append(data[target_name].to(DEVICE).unsqueeze(-1))
                task_targets = torch.cat(task_targets, dim=-1)
                
                task_loss_func = task2loss_func[task]
                if isinstance(task_loss_func, nn.CrossEntropyLoss):
                    task_targets = task_targets.squeeze(-1)

                out_range = task2output_range[task]
                task_outputs = outputs[:, out_range[0]:out_range[1]]

                task_loss = task_loss_func(task_outputs, task_targets)
                batch_loss_dict[task] = task_loss
                val_task_loss[task] += task_loss.item() * batch_size
            
            task_weight = 1 / len(task_list)
            batch_val_loss = [v.item() * task_weight for k, v in batch_loss_dict.items()]
            # used to calculate weighted loss specified by hand
            # batch_val_loss = [v.item() * task2weights[k] for k, v in batch_loss_dict.items()]
            val_loss += sum(batch_val_loss) * batch_size

            # Compute accuracy and accumulate
            if 'category_class' in task_list:
                out_range = task2output_range['category_class']
                _, predicted = torch.max(outputs[:, out_range[0]:out_range[1]], 1)
                category_label = data['category_label'].to(DEVICE)
                category_correct += (predicted == category_label).sum().item()
                
                # Log one batch of images to the dashboard, always same batch_idx.
                if i==batch_idx and log_images:
                    log_image_table(inputs, 
                                    predicted, 
                                    category_label, 
                                    outputs[:, out_range[0]:out_range[1]].softmax(dim=1),
                                    valid_dl.dataset.mappings['category_int2str'],
                                    )
            
            if 'object_class' in task_list:
                out_range = task2output_range['object_class']
                _, predicted = torch.max(outputs[:, out_range[0]:out_range[1]], 1)
                object_label = data['object_label'].to(DEVICE)
                object_correct += (predicted == object_label).sum().item()
            
            for task in task_list:
                if task in cat_reduced_tasks:
                    reduced_index = task[-1]
                    out_range = task2output_range[task]
                    _, predicted = torch.max(outputs[:, out_range[0]:out_range[1]], 1)
                    cat_red_label = data['cat_label_reduce' + reduced_index].to(DEVICE)
                    cat_red_correct += (predicted == cat_red_label).sum().item()
        
        return_dict = {}
        for task in task_list:
            return_dict[f'val_{task}_loss'] = val_task_loss[task] / image_ct
            
            if task in cat_reduced_tasks:
                return_dict['val_' + task + '_acc'] = cat_red_correct / image_ct
            
        if 'category_class' in task_list:
            return_dict['val_category_acc'] = category_correct / image_ct
        if 'object_class' in task_list:
            return_dict['val_object_acc'] = object_correct / image_ct

    return val_loss / image_ct, return_dict


def log_image_table(images, predicted, labels, probs, label2str):
    """
    Log a batch of data to wandb.Table
    img, pred, target, scores
    args:
        images: torch.Tensor of shape (batch_size, C, H, W)
        predicted: torch.Tensor of shape (batch_size,)
        labels: torch.Tensor of shape (batch_size,)
        probs: torch.Tensor of shape (batch_size, len(label2str))
        label2str: a dict mapping label int to label string
    """
    # Create a wandb Table to log images, labels and predictions to
    table = wandb.Table(columns=["image", "pred", "target"]
                        + [f"score_{label2str[i]}" for i in range(len(label2str))])

    images = images.detach().cpu().numpy()
    predicted = predicted.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    probs = probs.detach().cpu().numpy()

    for img, pred, targ, prob in zip(images, predicted, labels, probs):
        table.add_data(wandb.Image(img.transpose((1, 2, 0)) * 255), 
                       label2str[pred], 
                       label2str[targ], 
                       *prob)
    wandb.log({"predictions_table":table}, commit=False)


def train_slurm(config_path):
    config = load_config(config_path)
    train_model(config)


def train_model(config):
    wandb.init(
        project="multi-task-vision",
        config=config,
        # mode="disabled",
        )
    run_name = wandb.run.name if wandb.run.name else 'test'
    config = wandb.config

    # set up random seeds
    np.random.seed(NP_SEED + config.seed)
    torch.manual_seed(TCH_SEED + config.seed)

    start_time = datetime.now()
    
    assert config.max_batch % config.eval_per == 0
    
    # initialize the model
    if config.pretrain_init:
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = resnet18()

    output_number, task2output_range = get_output_info(config.dataset_name)

    # Replace the last layer with a linear layer for multi-task learning
    model.fc = nn.Linear(model.fc.in_features, output_number)
    model = model.to(DEVICE)

    model = torch.compile(model, fullgraph=True, dynamic=False)

    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # Get dataloaders
    train_loader = get_dataloader(dataset_name=config.dataset_name,
                                  is_train=True, 
                                  batch_size=config.batch_size,
                                  transform=IMN_transform,
                                  dataset_fraction=config.train_dataset_fraction)
    val_loader = get_dataloader(dataset_name=config.dataset_name,
                                is_train=False,
                                batch_size=config.batch_size,
                                transform=IMN_transform,
                                )

    # initialize
    batch_n = 0  # the numbder of batches the model has trained on so far
    sample_ct = 0  # number of training samples the model has trained on so far
    best_category_acc = 0.0
    best_object_acc = 0.0

    # restart from checkpoint if checkpoint exist
    checkpoint_path = Path(os.path.join(config.save_path, 'checkpoint.tar'))
    if checkpoint_path.is_file() and config.restart_from_checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        batch_n = checkpoint['batch_number']
        sample_ct = checkpoint['sample_count']
        best_category_acc = checkpoint['best_category_accuracy']
        best_object_acc = checkpoint['best_object_accuracy']
        print(f'Loaded checkpoint! Restarting from batch: {batch_n}')
    
    # Train the model
    model.train()
    while batch_n < config.max_batch:
        for data in train_loader:
            inputs = data['image'].to(DEVICE)
            outputs = model(inputs)

            task_loss_dict = {}
            for task in config.tasks:
                task_targets = []
                for target_name in task2targets_name[task]:
                    # data[target_name] is a tensor of shape (batch_size, )
                    task_targets.append(data[target_name].to(DEVICE).unsqueeze(-1))
                task_targets = torch.cat(task_targets, dim=-1)

                task_loss_func = task2loss_func[task]
                if isinstance(task_loss_func, nn.CrossEntropyLoss):
                    task_targets = task_targets.squeeze(-1)

                out_range = task2output_range[task]
                task_outputs = outputs[:, out_range[0]:out_range[1]]

                task_loss_dict[task] = task_loss_func(task_outputs, task_targets)
            
            task_weight = 1.0 / len(config.tasks)
            weighted_loss = [v * task_weight for k, v in task_loss_dict.items()]
            # used to calculate weighted loss specified by hand
            # weighted_loss = [v * task2weights[k] for k, v in task_loss_dict.items()]
            train_loss = 0.0
            for loss in weighted_loss:
                train_loss += loss
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            batch_n += 1
            sample_ct += len(inputs)

            metrics = {"train/batch_n": batch_n,
                       "train/sample_ct": sample_ct,
                       "train/train_loss": train_loss.item()}
            metrics.update({f"train/train_{k}_loss": v.item() for k, v in task_loss_dict.items()})
            
            # Log train metrics to wandb (last batch maybe smaller)
            if batch_n % config.eval_per != 0:
                wandb.log(metrics)
            # validate model
            else:
                val_loss, val_results = validate_model(model, val_loader, config.tasks, 
                                                       task2output_range,
                                                       log_images=(batch_n==config.max_batch))
                model.train()
                # Log train and validation metrics to wandb
                val_metrics = {"val/val_loss": val_loss}
                val_metrics.update({f"val/{k}": v for k, v in val_results.items()})
                wandb.log({**metrics, **val_metrics})

                out_string = f"Batch Number: {batch_n:10d}, Train Loss: {train_loss.item():.3f}, Valid Loss: {val_loss:.3f}"
                if 'category_class' in config.tasks:
                    category_acc = val_results['val_category_acc']
                    if category_acc > best_category_acc:
                        best_category_acc = category_acc
                    out_string += f", Valid Category Accuracy: {category_acc:.2f}"
                if 'object_class' in config.tasks:
                    object_acc = val_results['val_object_acc']
                    if object_acc > best_object_acc:
                        best_object_acc = object_acc
                    out_string += f", Valid Object Accuracy: {object_acc:.2f}"
                print(out_string)
            
            # save model checkpoint for restarting, at the very end of each loop
            if batch_n % config.checkpoint_per == 0:
                torch.save({
                    'batch_number': batch_n,
                    'sample_count': sample_ct,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_category_accuracy': best_category_acc,
                    'best_object_accuracy': best_object_acc,
                    }, checkpoint_path)
            
            if batch_n >= config.max_batch:
                break
    
    # wait 2.5 seconds
    time.sleep(2.5)

    # save the final model
    torch.save(model.state_dict(), os.path.join(config.save_path, 'model.pth'))

    # log summary metrics
    wandb.summary['best_category_accuracy'] = best_category_acc
    wandb.summary['best_object_accuracy'] = best_object_acc
    wandb.finish()
    
    log_complete(config.save_path, start_time)
    return model
