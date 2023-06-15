import numpy as np
import torch
from torchvision.models import resnet18, ResNet18_Weights
from collections import defaultdict
import torchvision.transforms as transforms

from dataset import HVMDataset
from config_global import DEVICE


def append_tuple(name, activity_dict, output):
    """
    append a tuple of output to activity_dict
    the names in the tuple are automatically generated as 
    name_1, name_2, name_3, ...
    args:
        name: the name of module to record activities
        activity_dict: a collection.defaultdict with default factory function set to list
            the activities will be stored in activity_dict[name]
        output: a tuple of torch.Tensor
    """
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
    Returns a hook function that can be registered with model layer
    to obtain and store the output history of hidden activations in activation_dict
    args:
        name: the name of module to record activities
        activity_dict: a collection.defaultdict with default factory function set to list
            the activities will be stored in activity_dict[name]
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


def get_activity(dataset, model, layers, remove_duplicates=lambda x: x):
    """
    get the activations of the model on the dataset
    args:
        dataset: a torch.utils.data.Dataset object,
            input of the model is accessed by dataset[i]['image']
        model: a torch.nn.Module object
        layers: a list of layer names to record activations
        remove_duplicates: a function that removes duplicate activations
            default is identity function that do nothing
    returns:
        all_activity: a dict of activations 
            for each specified layer (key in the dict)
            all_activity[layer_name] is a numpy array of shape
            (num_samples, num_neurons)
    """
    all_activity = defaultdict(list)

    for name, m in model.named_modules():
        if name in layers:
            m.register_forward_hook(append_activations(name, all_activity))

    model = model.to(DEVICE)
    model.eval()
    with torch.inference_mode():
        for i in range(len(dataset)):
            image = dataset[i]['image'].to(DEVICE)
            image = image.unsqueeze(0)
            _ignore = model(image)
            remove_duplicates(all_activity)

    for k, v in all_activity.items():
        activity = np.concatenate(v, axis=0)
        # reduce extra dimensions
        # so that the dimensions are (num_samples, num_neurons)
        all_activity[k] = np.reshape(activity, (activity.shape[0], -1))
        
    return all_activity


if __name__ == '__main__':

    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    dataset = HVMDataset(split='train', transform=transform)

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    record_layers = ['layer4.1.relu', 'avgpool']
    
    def remove_resnet_duplicates(activity_dict):
        # reduce the duplicate activations in resnet
        # because the later relu layer are used twice in resnet,
        for k, v in activity_dict.items():
            if '.relu' in k:
                v.pop(-2)
    remove_func = remove_resnet_duplicates

    activations = get_activity(dataset=dataset, model=model,
                               layers=record_layers,
                               remove_duplicates=remove_func)

    print(activations)