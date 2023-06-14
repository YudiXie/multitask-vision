import torch
from torchvision.models import resnet18, ResNet18_Weights
from collections import defaultdict
import torchvision.transforms as transforms

from dataset import HVMDataset

def append_tuple(name, activity_dict, output):
    for i_, otp in enumerate(output):
        new_name = name + '_' + str(i_ + 1)
        if isinstance(otp, tuple):
            append_tuple(new_name, activity_dict, otp)
        elif isinstance(otp, torch.Tensor):
            activity_dict[new_name].append(otp.detach().cpu().numpy())
        else:
            raise NotImplementedError('append type not implemented')


def append_activations(name, activity_dict):
    """
    Returns a hook function that can be registered with model layers
        to obtain and store the output history of hidden activations in activation_dict
    name: the name of module to record activities
    activity_dict: a collection.defaultdict with default factory function set to list
    """
    assert isinstance(activity_dict, defaultdict) \
        and activity_dict.default_factory == list, 'activity_dict must be default dict'

    def hook(module, inp, otp):
        if isinstance(otp, torch.Tensor):
            activity_dict[name].append(otp.detach().cpu().numpy())
        elif isinstance(otp, tuple):
            append_tuple(name, activity_dict, otp)
        else:
            raise NotImplementedError('append type not implemented')

    return hook


if __name__ == '__main__':
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    dataset = HVMDataset(split='train', transform=transform)


    layers = ['layer4.1.relu', 'avgpool']

    batch_activity = defaultdict(list)
    all_activity = defaultdict(list)

    for name, m in model.named_modules():
        if layers is None or name in layers:
            m.register_forward_hook(append_activations(name, batch_activity))

    for i in range(10):
        sample = dataset[i]
        image = sample['image']
        print(image)
        image = image.unsqueeze(0)
        output = model(image)
        for k, v in batch_activity.items():
            all_activity[k].append(np.concatenate(v, axis=0))
        batch_activity = defaultdict(list)
