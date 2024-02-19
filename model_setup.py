from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights

model_setup_dict = {
    'resnet18': resnet18,
    'resnet50': resnet50,
}

model_pretrain_weights = {
    'resnet18': ResNet18_Weights.IMAGENET1K_V1,
    'resnet50': ResNet50_Weights.IMAGENET1K_V1,
}
