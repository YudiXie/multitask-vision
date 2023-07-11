import numpy as np
import torch
from torchvision.models import resnet18, ResNet18_Weights, ResNet
from collections import defaultdict
import torchvision.transforms as transforms
from sklearn import linear_model, svm

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


def get_model_activations(dataset, model, layers, 
                          remove_duplicates=lambda x: x):
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


def get_model_activations_on_dataset(model, record_layers: list):
    """
    get the activations of the model on the dataset
    args:
        model: a torch.nn.Module object
        record_layers: a list of layer names to record activations
    returns:
        all_activations: a dict of activations
            each key is the region names
            each value is a numpy array of shape (num_images, num_neurons)
        data_frame: a pandas dataframe of the dataset
            that have num_images rows, each stores metadata of the stimulus
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

    dataset = HVMDataset(split='all', transform=transform)
    all_activations = get_model_activations(dataset, model, 
                                            record_layers, remove_func)
    return all_activations, dataset.normed_data_frame


def get_neural_activations(image_id_list, record_regions):
    """
    get the neural activations of the dataset
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


def get_neural_activations_on_dataset(record_regions: list):
    """
    get the neural activations of the dataset
    args:
        record_regions: a list of region names to record activities
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
    all_activations = get_neural_activations(imgid_list, record_regions)
    return all_activations, data_frame


def create_train_test_split(all_activity, data_frame, target_name,
                            train_ratio=0.8):
    """
    create train and test split
    args:
        all_activity: a numpy array of shape (num_all_images, num_neurons)
        data_frame: a pandas dataframe of the dataset
            that have num_images rows, each stores metadata of the stimulus
        target_name: a string of target name, eg. 's'
        train_ratio: float, the ratio of train data in the whole dataset
    returns:
        data: a dict of train and test activities and targets
            data['train_activity'], data['test_activity']:
                a numpy array of shape (num_train/test_images, num_neurons)
            data['train_target'], data['test_target']: 
                a numpy array of shape (num_train/test_images,)
    """
    # create train and test split
    # train 4608 images, test 1152 images if train_ratio=0.8 and total 5760 images
    assert all_activity.shape[0] == len(data_frame), 'number of images must match'
    data_len = len(data_frame)
    permuted_index = np.random.permutation(data_len)
    train_len = int(data_len * train_ratio)
    train_index = permuted_index[:train_len]
    test_index = permuted_index[train_len:]

    train_activity = all_activity[train_index, :]
    test_activity = all_activity[test_index, :]
    
    all_target = data_frame[target_name].to_numpy(copy=True)
    train_target = all_target[train_index]
    test_target = all_target[test_index]

    data = {}
    data['train_activity'] = train_activity
    data['test_activity'] = test_activity
    data['train_target'] = train_target
    data['test_target'] = test_target
    return data


def evaluate_regression(train_activity, test_activity, 
                        train_target, test_target):
    """
    evaluate the regression model on the dataset
    args:
        train_activity: ndarray of shape (num_train_images, num_neurons)
        test_activity: ndarray of shape (num_test_images, num_neurons)
        train_target: ndarray of shape (num_train_images,), targets are continuous
        test_target: ndarray of shape (num_test_images,), targets are continuous
    returns:
        correlation coefficient, p-value
    """
    # fit regression model
    alphas = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, .75e-1, 1e0, 2.5e0, 5e0, 1e1, 25, 1e2, 1e3]
    reg = linear_model.RidgeCV(alphas=alphas).fit(train_activity, train_target)
    # print(reg.score(X, y))
    return stats.pearsonr(reg.predict(test_activity), test_target)


def evaluate_classification(train_activity, test_activity, 
                            train_target, test_target):
    """
    evaluate the regression model on the dataset
    args:
        train_activity: ndarray of shape (num_train_images, num_neurons)
        test_activity: ndarray of shape (num_test_images, num_neurons)
        train_target: ndarray of shape (num_train_images,), targets are discrete class labels
            like [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        test_target: ndarray of shape (num_test_images,), targets are discrete class labels
            like [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    returns:
        evaluation accuracy
    """
    # fit classification model
    clf = svm.LinearSVC(C=5e-3)
    clf.fit(train_activity, train_target)
    return clf.score(test_activity, test_target)


def cross_validate_on_target(activity, df, target_name,
                             downsample_number=128,
                             num_cross_val=30,
                             mode='regression'
                             ):
    """
    cross validate the regression model on the dataset
    args:
        activity: a numpy array of shape (num_all_images, num_neurons)
        df: a pandas dataframe of the dataset
            that have num_images rows, each stores metadata of the stimulus
        target_name: a string of target name, eg. 's'
        downsample_number: int, number of neurons to downsample to
        num_cross_val: int, number of cross validation
        mode: string, 'regression' or 'classification'
    returns:
        for regression mode:
        mean correlation coefficient, std of correlation coefficient
        for classification mode:
        mean accuracy, std of accuracy
    """
    assert mode in ['regression', 'classification']

    performance_list = []
    for i in range(num_cross_val):
        train_test_data = create_train_test_split(activity, df, target_name)
        
        # downsample neurons
        if downsample_number is not None:
            assert train_test_data['train_activity'].shape[1] == train_test_data['test_activity'].shape[1]
            num_neurons = train_test_data['train_activity'].shape[1]
            sample_ids = np.random.choice(num_neurons, downsample_number, replace=False)
            train_test_data['train_activity'] = train_test_data['train_activity'][:, sample_ids]
            train_test_data['test_activity'] = train_test_data['test_activity'][:, sample_ids]

        if mode == 'regression':
            coef, pval = evaluate_regression(**train_test_data)
            performance_list.append(coef)
        else:
            acc = evaluate_classification(**train_test_data)
            performance_list.append(acc)

    return np.mean(performance_list), np.std(performance_list)


def target_direction_vector(activity, targets):
    """
    get dirctional vector of the target variable
    args:
        activity: a numpy array of shape (num_all_images, num_neurons)
        targets: a numpy array of shape (num_all_images,) for a target value eg. 's'
    returns:
        a vector containing regression coefficients of the target variable
    """
    # TODO test regularization
    # TODO test PCA first
    # TODO test normalization

    # center activity
    activity = activity[:, :100]
    activity = activity - np.mean(activity, axis=0)

    reg = linear_model.LinearRegression().fit(activity, targets)
    return reg.coef_


if __name__ == '__main__':
    pass
