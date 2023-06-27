import numpy as np
import torch
from torchvision.models import resnet18, ResNet18_Weights, ResNet
from collections import defaultdict
import torchvision.transforms as transforms
from sklearn import linear_model

from dataset import HVMDataset
from config_global import DEVICE
from scipy import stats

from brainio import get_assembly, get_stimulus_set


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


def get_neural_activity(image_id_list, record_regions):
    """
    get the neural activity of the dataset
    args:
        image_id_list: a list of image id to record activations
        record_regions: a list of region names to record activations
            eg. ['V4', 'IT']
    returns:
        activations: a dict of activations
            each key is the region names
            each value is a numpy array of shape (num_images, num_neurons)
                that are the activations of the images in image_id_list
    """
    # 128 V4 neurons, 168 IT neurons, 5760 images
    # (296, 268800, 1) arrary (neuroid, presentation, time_bin)
    hvm_assy = get_assembly(identifier="dicarlo.MajajHong2015")
    # (296, 5760, 1) arrary (neuroid, presentation, time_bin) mean over repetitions
    dataassy_mean = hvm_assy.groupby('stimulus_id').mean()

    activations = {}
    for rg in record_regions:
        region_act = dataassy_mean.sel(region=rg, stimulus_id=image_id_list).values
        activations[rg] = region_act.squeeze().transpose()
    return activations


def get_neural_activity_on_dataset(record_regions: list):
    """
    get the neural activations of the dataset
    args:
        record_regions: a list of region names to record activations
    returns:
        all_activations: a dict of activations
            each key is the region names
            each value is a numpy array of shape (num_images, num_neurons)
        data_frame: a pandas dataframe of the dataset
            that have num_images rows, each stores metadata of the stimulus
    """
    # all neural data 5760 images in total
    dataset = HVMDataset(split='all')
    data_frame = dataset.normed_data_frame
    imgid_list = list(data_frame['image_id'])
    all_activations = get_neural_activity(imgid_list, record_regions)
    return all_activations, data_frame


def create_train_test_split(all_activations, data_frame, target_name,
                            train_ratio=0.8):
    """
    create train and test split
    args:
        all_activations: a dict of activations
            each key is the region names
            each value is a numpy array of shape (num_images, num_neurons)
        data_frame: a pandas dataframe of the dataset
            that have num_images rows, each stores metadata of the stimulus
        target_name: a string of target to record activations, eg. 's'
        train_ratio: float, the ratio of train data in the whole dataset
    returns:
        data: a dict of train and test activations and targets
            data['train_activations'], data['test_activations']: a dict of activations
                each key is the region names
                each value is a numpy array of shape (num_images, num_neurons)
            data['train_target'], data['test_target']: a numpy array of shape (num_images,)
    """
    # create train and test split
    # train 4608 images, test 1152 images
    data_len = len(data_frame)
    permuted_index = np.random.permutation(data_len)
    train_len = int(data_len * train_ratio)
    train_index = permuted_index[:train_len]
    test_index = permuted_index[train_len:]

    train_activations = {}
    test_activations = {}
    for region, activity in all_activations.items():
        train_activations[region] = activity[train_index, :]
        test_activations[region] = activity[test_index, :]
    
    all_target = data_frame[target_name].to_numpy(copy=True)
    train_target = all_target[train_index]
    test_target = all_target[test_index]

    data = {}
    data['train_activations'] = train_activations
    data['test_activations'] = test_activations
    data['train_target'] = train_target
    data['test_target'] = test_target
    return data


def evaluate_regression(train_activity, test_activity, 
                        train_target, test_target,
                        downsample_number=None):
    """
    evaluate the regression model on the dataset
    args:
        train_activity: ndarray of shape (num_train_images, num_neurons)
        test_activity: ndarray of shape (num_test_images, num_neurons)
        train_target: ndarray of shape (num_train_images,)
        test_target: ndarray of shape (num_test_images,)
        downsample_number: int, number of neurons to downsample to
    returns:
        correlation coefficient, p-value
    """
    assert train_activity.shape[1] == test_activity.shape[1]
    num_neurons = train_activity.shape[1]

    if downsample_number is not None:
        print(f'Downsampling to have {downsample_number} neurons')
        sample_ids = np.random.choice(num_neurons, downsample_number, replace=False)
        train_activity = train_activity[:, sample_ids]
        test_activity = test_activity[:, sample_ids]

    # fit regression model
    alphas = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, .75e-1, 1e0, 2.5e0, 5e0, 1e1, 25, 1e2, 1e3]
    reg = linear_model.RidgeCV(alphas=alphas).fit(train_activity, train_target)
    # print(reg.score(X, y))
    return stats.pearsonr(reg.predict(test_activity), test_target)


def evaluate_regression_on_region(region, data):
    return evaluate_regression(data['train_activations'][region],
                               data['test_activations'][region],
                               data['train_target'],
                               data['test_target'])


if __name__ == '__main__':
    pass
    # model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # data = get_activity_on_dataset(model, ['layer3.1.relu', 'avgpool'])
    # evaluate_regression('layer3.1.relu', 's', **data)
    # evaluate_regression('avgpool', 's', **data)
