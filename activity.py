import numpy as np
import torch
from torchvision.models import resnet18, ResNet18_Weights, ResNet
from collections import defaultdict
import torchvision.transforms as transforms
from sklearn.linear_model import LinearRegression

from dataset import HVMDataset
from config_global import DEVICE
from scipy import stats

from brainio import get_assembly, get_stimulus_set
from tqdm import tqdm


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
    handles = defaultdict(list)

    for name, m in model.named_modules():
        if name in layers:
            handles[name] = m.register_forward_hook(append_activations(name, all_activity))

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
    
    # remove hooks
    for k, v in handles.items():
        v.remove()
        
    return all_activity


def get_activity_on_dataset(model, record_layers: list):
    """
    get the activations of the model on the dataset
    args:
        model: a torch.nn.Module object
        record_layers: a list of layer names to record activations
    returns:
        data: a dict of activations and datasets
    """

    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    assert isinstance(model, ResNet), 'model must be a resnet'
    def remove_resnet_duplicates(activity_dict):
        # reduce the duplicate activations in resnet
        # because the later relu layer are used twice in resnet,
        for k, v in activity_dict.items():
            if '.relu' in k:
                v.pop(-2)
    remove_func = remove_resnet_duplicates

    # train data
    train_dataset = HVMDataset(split='train', transform=transform)
    train_activations = get_activity(dataset=train_dataset, model=model,
                                     layers=record_layers,
                                     remove_duplicates=remove_func)

    # validation data
    val_dataset = HVMDataset(split='val', transform=transform)
    val_activations = get_activity(dataset=val_dataset, model=model,
                                   layers=record_layers,
                                   remove_duplicates=remove_func)
    
    data = {}
    data['train_activations'] = train_activations
    data['val_activations'] = val_activations
    data['train_dataset'] = train_dataset
    data['val_dataset'] = val_dataset
    
    return data


def get_neural_activity(dataset, record_layers):
    hvm_assy = get_assembly(identifier="dicarlo.MajajHong2015")
    dataassy_mean = hvm_assy.groupby('stimulus_id').mean()

    activations = defaultdict(list)
    image_id_list = dataset.normed_data_frame['image_id']
    for layer in record_layers:
        print(f'getting data for layer: {layer}')
        for image_id in tqdm(image_id_list):
            act = dataassy_mean.sel(region=layer, stimulus_id=image_id).values.squeeze()
            activations[layer].append(act)
        activations[layer] = np.stack(activations[layer], axis=0) 

    return activations


def get_neural_activity_on_dataset(record_layers: list):
    """
    get the neural activations of the dataset
    args:
        record_layers: a list of layer names to record activations
    returns:
        data: a dict of activations and datasets
    """

    # train data
    train_dataset = HVMDataset(split='train')
    train_activations = get_neural_activity(train_dataset, record_layers)

    # validation data
    val_dataset = HVMDataset(split='val')
    val_activations = get_neural_activity(val_dataset, record_layers)
    
    data = {}
    data['train_activations'] = train_activations
    data['val_activations'] = val_activations
    data['train_dataset'] = train_dataset
    data['val_dataset'] = val_dataset
    
    return data


def evaluate_regression(layer, target, 
                        train_activations, val_activations, 
                        train_dataset, val_dataset):
    X_train = train_activations[layer]
    y_train = np.array(train_dataset.normed_data_frame[target])

    # fit regression model
    reg = LinearRegression().fit(X_train, y_train)
    # print(reg.score(X, y))

    X_val = val_activations[layer]
    y_val = np.array(val_dataset.normed_data_frame[target])

    return stats.pearsonr(reg.predict(X_val), y_val)


if __name__ == '__main__':
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    data = get_activity_on_dataset(model, ['layer3.1.relu', 'avgpool'])
    evaluate_regression('layer3.1.relu', 's', **data)
    evaluate_regression('avgpool', 's', **data)
