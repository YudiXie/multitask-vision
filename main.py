import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

from dataset import HVMDataset

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # Replace the last layer with a linear layer for ImageNet classification
    # num_objects = 8
    # num_categories = 8
    # num_predict = 6
    # num_output = num_objects + num_categories + num_predict
    model.fc = nn.Linear(model.fc.in_features, 8)
    model = model.to(device)

    # Data preprocessing
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # Download and load the ImageNet dataset
    train_dataset = HVMDataset(split='train', transform=transform)
    val_dataset = HVMDataset(split='val', transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train the model
    model.train()
    i_batch = 0
    max_batch = 10000
    eval_per = 10
    best_acc = 0.0

    while i_batch <= max_batch:
        for data in train_loader:
            print(f'Batch: {i_batch}')

            if i_batch % eval_per == 0:
                model.eval()
                with torch.no_grad():
                    running_loss = 0.0
                    running_corrects = 0
                    for data in val_loader:
                        inputs = data['image'].to(device)
                        labels = data['category_label'].to(device)

                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                    
                    val_loss = running_loss / len(val_dataset)
                    val_acc = running_corrects / len(val_dataset)
                    if val_acc >= best_acc:
                        best_acc = val_acc
                        torch.save(model.state_dict(), 'best_model.pth')
                    
                    print(f'Val loss: {val_loss}')
                    print(f'Val acc: {val_acc}')

                model.train()

            inputs = data['image'].to(device)
            labels = data['category_label'].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            i_batch += 1
