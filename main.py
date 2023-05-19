import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import wandb

from dataset import HVMDataset


def get_dataloader(is_train, batch_size):
    "Get a training dataloader"
    # Data preprocessing
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    if is_train:
        split = 'train'
    else:
        split = 'val'

    dataset = HVMDataset(split=split, transform=transform)
    loader = torch.utils.data.DataLoader(dataset=dataset, 
                                         batch_size=batch_size, 
                                         shuffle=True if is_train else False, 
                                         pin_memory=True, 
                                         num_workers=2)
    return loader


# TODO: need to check if this is correct
def validate_model(model, valid_dl, loss_func, log_images=False, batch_idx=0):
    "Compute performance of the model on the validation dataset and log a wandb.Table"
    val_loss = 0.0
    correct = 0
    model.eval()
    with torch.inference_mode():
        for i, data in enumerate(valid_dl):
            inputs = data['image'].to(DEVICE)
            labels = data['category_label'].to(DEVICE)

            # Forward pass ➡
            outputs = model(inputs)
            val_loss += loss_func(outputs, labels).item() * labels.size(0)

            # Compute accuracy and accumulate
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            # Log one batch of images to the dashboard, always same batch_idx.
            if i==batch_idx and log_images:
                log_image_table(inputs, predicted, labels, outputs.softmax(dim=1))
    return val_loss / len(valid_dl.dataset), correct / len(valid_dl.dataset)


# TODO: need to check if this is correct
def log_image_table(images, predicted, labels, probs):
    "Log a wandb.Table with (img, pred, target, scores)"
    # 🐝 Create a wandb Table to log images, labels and predictions to
    table = wandb.Table(columns=["image", "pred", "target"]+[f"score_{i}" for i in range(10)])
    for img, pred, targ, prob in zip(images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):
        table.add_data(wandb.Image(img[0].numpy()*255), pred, targ, *prob.numpy())
    wandb.log({"predictions_table":table}, commit=False)


if __name__ == '__main__':
    wandb.init(
        project="multi-task-vision",
        config={
            "batch_size": 32,
            "lr": 1e-3,
            "max_batch": 500,
            "eval_per": 10,
            })
    
    # Copy your config 
    config = wandb.config
    assert config.max_batch % config.eval_per == 0

    # device to run algorithm on
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
    
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # Replace the last layer with a linear layer for ImageNet classification
    # num_objects = 8
    # num_categories = 8
    # num_predict = 6
    # num_output = num_objects + num_categories + num_predict
    model.fc = nn.Linear(model.fc.in_features, 8)
    model = model.to(DEVICE)

    train_loader = get_dataloader(is_train=True, batch_size=config.batch_size)
    val_loader = get_dataloader(is_train=False, batch_size=config.batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # Train the model
    model.train()
    batch_n = 1
    example_ct = 0
    best_acc = 0.0
    while batch_n < config.max_batch + 1:
        for data in train_loader:
            inputs = data['image'].to(DEVICE)
            labels = data['category_label'].to(DEVICE)

            outputs = model(inputs)
            train_loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            example_ct += len(inputs)
            metrics = {"train/batch_n": batch_n,
                       "train/example_ct": example_ct,
                       "train/train_loss": train_loss.item()}
            
            # Log train metrics to wandb (last batch maybe smaller)
            if batch_n % config.eval_per != 0:
                wandb.log(metrics)
            # validate model
            else:
                val_loss, accuracy = validate_model(model, val_loader, criterion, 
                                                    log_images=(batch_n==config.max_batch))
                model.train()
                # Log train and validation metrics to wandb
                val_metrics = {"val/val_loss": val_loss, 
                               "val/val_accuracy": accuracy}
                wandb.log({**metrics, **val_metrics})
                print(f"Batch Number: {batch_n:4d}, Train Loss: {train_loss:.3f}, Valid Loss: {val_loss:3f}, Valid Accuracy: {accuracy:.2f}")

                # save best model
                if accuracy >= best_acc:
                    best_acc = accuracy
                    torch.save(model.state_dict(), 'best_model.pth')

            batch_n += 1
            if batch_n > config.max_batch:
                break
    
    # log a Summary metric
    # wandb.summary['test_accuracy'] = 0.8
    wandb.finish()
