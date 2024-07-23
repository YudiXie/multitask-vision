from brainio import get_assembly, get_stimulus_set
from dataset import HVMDataset


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
