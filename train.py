import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import wandb

from config_global import DEVICE, NP_SEED, TCH_SEED
from dataset import HVMDataset
from utils import load_config


task2targets_name = {
    'category_class': ['category_label'],
    'object_class': ['object_label'],
    'rotation_reg': ['rxy', 'rxz', 'ryz'],
    'size_reg': ['s'],
    'translation_reg': ['ty', 'tz'],
}

task2loss_func = {
    'category_class': nn.CrossEntropyLoss(),
    'object_class': nn.CrossEntropyLoss(),
    'rotation_reg': nn.MSELoss(),
    'size_reg': nn.MSELoss(),
    'translation_reg': nn.MSELoss(),
}

task2output_range = {
    'category_class': (0, 8),
    'object_class': (8, 72),
    'rotation_reg': [72, 75],
    'size_reg': [75, 76],
    'translation_reg': [76, 78],
}

task2weights = {
    'category_class': 0.2,
    'object_class': 0.2,
    'rotation_reg': 0.2,
    'size_reg': 0.2,
    'translation_reg': 0.2,
}


def log_complete(exp_path: str, start_time=None):
    """
    create a file to indicate the training is finished
    """
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    
    complete_time = datetime.now()
    with open(os.path.join(exp_path, 'train_complete.txt'), 'w') as f:
        f.write(f'Training is complete at: {complete_time.strftime("%Y-%m-%d %H:%M:%S")}')
        if start_time is not None:
            f.write(f'\nTraining time: {str(complete_time - start_time)}')
    
    print(f'Training is complete at: {complete_time.strftime("%Y-%m-%d %H:%M:%S")}')


def get_dataloader(is_train, batch_size, transform):
    "Get a training dataloader"    
    if is_train:
        split = 'train'
    else:
        split = 'val'

    dataset = HVMDataset(split=split, transform=transform)
    loader = torch.utils.data.DataLoader(dataset=dataset, 
                                         batch_size=batch_size, 
                                         shuffle=True if is_train else False, 
                                         pin_memory=True, 
                                         num_workers=4)
    return loader


def validate_model(model,
                   valid_dl,
                   task_list,
                   log_images=False,
                   batch_idx=0):
    "Compute performance of the model on the validation dataset and log a wandb.Table"
    val_loss = 0.0
    val_task_loss = {task: 0.0 for task in task_list}
    category_correct = 0
    object_correct = 0
    image_ct = 0
    model.eval()
    with torch.inference_mode():
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
                if task[-5:] == 'class':
                    task_targets = task_targets.squeeze(-1)

                out_range = task2output_range[task]
                task_outputs = outputs[:, out_range[0]:out_range[1]]
                task_loss = task2loss_func[task](task_outputs, task_targets)
                batch_loss_dict[task] = task_loss
                val_task_loss[task] += task_loss.item() * batch_size
            
            batch_val_loss = [v.item() * task2weights[k] for k, v in batch_loss_dict.items()]
            val_loss += sum(batch_val_loss) * batch_size

            # Compute accuracy and accumulate
            if 'category_class' in task_list:
                _, predicted = torch.max(outputs[:, 0:8], 1)
                category_label = data['category_label'].to(DEVICE)
                category_correct += (predicted == category_label).sum().item()
                
                # Log one batch of images to the dashboard, always same batch_idx.
                if i==batch_idx and log_images:
                    log_image_table(inputs, 
                                    predicted, 
                                    category_label, 
                                    outputs[:, 0:8].softmax(dim=1),
                                    valid_dl.dataset.category_int2str,
                                    )
            
            if 'object_class' in task_list:
                _, predicted = torch.max(outputs[:, 8:72], 1)
                object_label = data['object_label'].to(DEVICE)
                object_correct += (predicted == object_label).sum().item()
        
        return_dict = {}
        for task in task_list:
            return_dict[f'val_{task}_loss'] = val_task_loss[task] / image_ct
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
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # Replace the last layer with a linear layer for multi-task learning
    model.fc = nn.Linear(model.fc.in_features, 78)
    model = model.to(DEVICE)

    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # Get dataloaders
    train_loader = get_dataloader(is_train=True, batch_size=config.batch_size,
                                  transform=transform)
    val_loader = get_dataloader(is_train=False, batch_size=config.batch_size,
                                transform=transform)

    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # Train the model
    model.train()
    batch_n = 1
    example_ct = 0
    best_category_acc = 0.0
    best_object_acc = 0.0
    while batch_n < config.max_batch + 1:
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
                if task[-5:] == 'class':
                    task_targets = task_targets.squeeze(-1)

                out_range = task2output_range[task]
                task_outputs = outputs[:, out_range[0]:out_range[1]]
                task_loss_dict[task] = task2loss_func[task](task_outputs, task_targets)
            
            weighted_loss = [v * task2weights[k] for k, v in task_loss_dict.items()]
            train_loss = 0.0
            for loss in weighted_loss:
                train_loss += loss
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            example_ct += len(inputs)
            metrics = {"train/batch_n": batch_n,
                       "train/example_ct": example_ct,
                       "train/train_loss": train_loss.item()}
            metrics.update({f"train/train_{k}_loss": v.item() for k, v in task_loss_dict.items()})
            
            # Log train metrics to wandb (last batch maybe smaller)
            if batch_n % config.eval_per != 0:
                wandb.log(metrics)
            # validate model
            else:
                val_loss, val_results = validate_model(model, val_loader, config.tasks,
                                                       log_images=(batch_n==config.max_batch))
                model.train()
                # Log train and validation metrics to wandb
                val_metrics = {"val/val_loss": val_loss}
                val_metrics.update({f"val/{k}": v for k, v in val_results.items()})
                wandb.log({**metrics, **val_metrics})

                out_string = f"Batch Number: {batch_n:4d}, Train Loss: {train_loss:.3f}, Valid Loss: {val_loss:.3f}"
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

            batch_n += 1
            if batch_n > config.max_batch:
                break
    
    # save the model
    torch.save(model.state_dict(), os.path.join(config.save_path, 'model.pth'))

    # log a Summary metric
    log_complete(config.save_path, start_time)
    wandb.summary['best_category_accuracy'] = best_category_acc
    wandb.summary['best_object_accuracy'] = best_object_acc
    wandb.alert(
            title='Run Finished',
            text=f'Run Finished, Best Category Accuracy: {best_category_acc:.2f}'
        )
    wandb.finish()
    return model
