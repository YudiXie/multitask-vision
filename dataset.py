
import os
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy.stats import circmean
from PIL import Image
import torch
from torch.utils.data import Dataset
from skimage import io
from tqdm import tqdm
import yaml


def center_circ_array(arr):
    """
    Center an array of circular data.
    arr: array of circular data, 1d numpy array, in degrees, range (-inf, inf)
    return: centered array, 1d numpy array, in degrees, range (-180, 180), 
        centered around 0, has circular mean of 0 (mean is not 0)
    """
    arr = arr % 360.0
    arr = arr - circmean(arr, high=360.0, low=0.0)
    arr = arr % 360.0
    arr[arr > 180.0] -= 360.0
    return arr


def create_mapping(str_list: list):
    """
    Create a mapping from string to int and int to string
    param: str_list: list of strings with no duplicates
    return: two dictionaries that contains the mapping from string to int and int to string
    """
    str_list.sort()
    str2int_map = {}
    for i, str_name in enumerate(str_list):
        str2int_map[str_name] = i
    int2str_map = {v: k for k, v in str2int_map.items()}
    return str2int_map, int2str_map


def load_image(image_filepath):
    """Load an image from disk and return a PIL.Image object.
    from https://github.com/brain-score/model-tools/blob/75365b54670d3f6f63dcdf88395c0a07d6b286fc/model_tools/activations/pytorch.py#L118
    """
    with Image.open(image_filepath) as pil_image:
        if 'L' not in pil_image.mode.upper() and 'A' not in pil_image.mode.upper() \
                and 'P' not in pil_image.mode.upper():  # not binary and not alpha and not palletized
            # work around to https://github.com/python-pillow/Pillow/issues/1144,
            # see https://stackoverflow.com/a/30376272/2225200
            return pil_image.copy()
        else:  # make sure potential binary images are in RGB
            rgb_image = Image.new("RGB", pil_image.size)
            rgb_image.paste(pil_image)
            return rgb_image



class HVMDataset(Dataset):
    """hvm-public dataset."""
    # images are 256x256 pixels
    # The axis labeling here uses the convention that
    #     +x coming "out of the screen"
    #     +z is "up" (vertical height) and
    #     +y is "right" (horizontal extent)
    
    # s (size)
    # tz, vertical translation (up + /down -)
    # ty, horizontal translation (right + /left -)
    # 'rxz', 'rxy', 'ryz', rotational parameters
    # 'r[xz/xy/yz]_semantic', semantically consistent rotational parameters,

    # all collums in the stimulus set
    all_collums = ['id', 'background_id', 's', 'image_id', 'image_file_name', 'filename', 'rxy', 'tz', 'category_name', 'rxz_semantic', 'ty', 'ryz', 'object_name', 'variation', 'size', 'rxy_semantic', 'ryz_semantic', 'rxz']
    # collums that need to be normalized
    norm_collums = ['s', 'ty', 'tz', 'rxy', 'rxz', 'ryz', 'rxy_semantic', 'rxz_semantic', 'ryz_semantic']

    def __init__(self, 
                 csv_file='./data/hvm_dataset/image_dicarlo_hvm.csv', 
                 root_dir='./data/hvm_dataset/image_dicarlo_hvm', 
                 split='train',
                 transform=None,
                 ):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        data_frame = pd.read_csv(csv_file)
        normed_data_frame = data_frame.copy()
        self.root_dir = root_dir
        self.transform = transform
      
        # create a map from category name to category label
        category_str2int, category_int2str = create_mapping(list(normed_data_frame['category_name'].unique()))
        normed_data_frame['category_label'] = [category_str2int[cn] for cn in normed_data_frame['category_name']]
        
        # create a map from object name to object label
        object_str2int, object_int2str = create_mapping(list(normed_data_frame['object_name'].unique()))
        normed_data_frame['object_label'] = [object_str2int[on] for on in normed_data_frame['object_name']]

        self.mappings = {
            'category_str2int': category_str2int,
            'category_int2str': category_int2str,
            'object_str2int': object_str2int,
            'object_int2str': object_int2str,
        }

        # normalize the data
        for collum in self.norm_collums:
            normed_data_frame[collum] = (data_frame[collum] - data_frame[collum].mean()) / data_frame[collum].std()
            normed_data_frame[collum] = normed_data_frame[collum].astype(np.float32)
        
        if split == 'all':
            self.normed_data_frame = normed_data_frame
        elif split == 'train':
            self.normed_data_frame = normed_data_frame[:int(len(normed_data_frame) * 0.8)]
        elif split == 'val':
            self.normed_data_frame = normed_data_frame[int(len(normed_data_frame) * 0.8):].reset_index(drop=True)
        else:
            raise ValueError('split must be either all, train, or val')

    def __len__(self):
        return len(self.normed_data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.normed_data_frame.loc[idx, 'image_file_name'])
        image = load_image(img_name)
        if self.transform:
            image = self.transform(image)
        sample = {'image': image}
        sample['category_label'] = self.normed_data_frame.loc[idx, 'category_label']
        sample['object_label'] = self.normed_data_frame.loc[idx, 'object_label']
        
        for i in range(2, 9):
            # reduce the 8 category labels (0..7) to 2, 3, 4, 5, 6, 7, 8 category labels
            sample[f'cat_label_reduce{i}'] = sample['category_label'] if sample['category_label'] < i else (i - 1)

        for collum in self.norm_collums:
            sample[collum] = self.normed_data_frame.loc[idx, collum]

        return sample


class TDWDataset(Dataset):
    """TDW dataset.
    images are 256x256 pixels
    neg_x: distance of object from the screen, in TDW world-space units, where the screen is at x = 0, + is going into the image
    ty: horizontal position of object, in pixels, center of image is 0, + is going right
    tz: vertical position of object, in pixels, center of image is 0, + is going up
    euler_1, euler_2, euler_3: rotation of object, in degrees, returned by TDW local transform relative to the camera
    euler_x_proc: rotation of object, in degrees, processed to be in the range (-180, 180), centered around 0

    tdw_image_dataset_small_multi_env(_hdri): 8 categories, ~5,000 images
    tdw_image_dataset_large_20230907: 117 categories, 587 objects, ~1,350,000 images
    """
    norm_collums = ['neg_x', 'ty', 'tz','euler_1_proc', 'euler_2_proc', 'euler_3_proc']

    def __init__(self, 
                 root_dir='./data/tdw_image_dataset_small_multi_env_hdri',
                 split='train',
                 transform=None,
                 ):
        """
        Arguments:
            root_dir (string): Directory with all the images.
        """
        assert Path(root_dir).joinpath('images_meta_shuffled_processed.csv').is_file(), \
            f'images_meta_shuffled_processed.csv does not exist in {root_dir}, please run tdw_dataset_preprocess() first'
        assert Path(root_dir).joinpath('mappings.yml').is_file(), \
            f'mappings.yml does not exist in {root_dir}, please run tdw_dataset_preprocess() first'
        
        normed_data_frame = pd.read_csv(os.path.join(root_dir, 'images_meta_shuffled_processed.csv'), index_col=0)
        for collum in TDWDataset.norm_collums:
            normed_data_frame[collum] = normed_data_frame[collum].astype(np.float32)

        dataset_name = os.path.basename(root_dir)
        if dataset_name == 'tdw_image_dataset_small_multi_env_hdri':
            split_index = int(len(normed_data_frame) * 0.8)
        elif dataset_name == 'tdw_image_dataset_large_20230907':
            split_index = -50000
        else:
            raise NotImplementedError(f'Dataset: {dataset_name} not implemented')
        
        self.root_dir = root_dir
        self.transform = transform

        with open(os.path.join(root_dir, 'mappings.yml'), 'r') as file:
            mappings = yaml.safe_load(file)
        self.mappings = mappings
        
        if split == 'all':
            self.normed_data_frame = normed_data_frame
        elif split == 'train':
            self.normed_data_frame = normed_data_frame[:split_index]
        elif split == 'val':
            self.normed_data_frame = normed_data_frame[split_index:].reset_index(drop=True)
        else:
            raise ValueError('split must be either all, train, or val')

    def __len__(self):
        return len(self.normed_data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_file = os.path.join(self.root_dir, self.normed_data_frame.loc[idx, 'image_filename'])
        image = load_image(img_file)
        if self.transform:
            image = self.transform(image)
        sample = {'image': image}
        sample['category_label'] = self.normed_data_frame.loc[idx, 'category_label']
        sample['object_label'] = self.normed_data_frame.loc[idx, 'object_label']
        
        for i in range(2, 9):
            # reduce the 8 category labels (0..7) to 2, 3, 4, 5, 6, 7, 8 category labels
            sample[f'cat_label_reduce{i}'] = sample['category_label'] if sample['category_label'] < i else (i - 1)

        for collum in self.norm_collums:
            sample[collum] = self.normed_data_frame.loc[idx, collum]

        return sample


def tdw_dataset_preprocess(root_dir):
    """
    Preprocess the TDW dataset.
    """
    csv_path = os.path.join(root_dir, 'images_meta_shuffled.csv')
    data_frame = pd.read_csv(csv_path, index_col=0)
    # shuffle the data
    # data_frame = data_frame.sample(frac=1, random_state=RNG).reset_index(drop=True)

    # check if all images exist
    data_path = Path(root_dir)
    for i in tqdm(range(len(data_frame))):
        assert data_path.joinpath(data_frame.loc[i, 'image_filename']).is_file()
    print(f'All images in csv file {csv_path} exist')

    normed_data_frame = data_frame.copy()

    # create a map from category name to category label
    category_str2int, category_int2str = create_mapping(list(normed_data_frame['wnid'].unique()))
    normed_data_frame['category_label'] = [category_str2int[cn] for cn in normed_data_frame['wnid']]
    
    # create a map from object name to object label
    object_str2int, object_int2str = create_mapping(list(normed_data_frame['record_name'].unique()))
    normed_data_frame['object_label'] = [object_str2int[on] for on in normed_data_frame['record_name']]

    mappings = {
        'category_str2int': category_str2int,
        'category_int2str': category_int2str,
        'object_str2int': object_str2int,
        'object_int2str': object_int2str,
    }

    # process the euler angles, center them around 0, and in range (-180, 180)
    data_frame['euler_1_proc'] = center_circ_array(data_frame['euler_1'].to_numpy())
    data_frame['euler_2_proc'] = center_circ_array(data_frame['euler_2'].to_numpy())
    data_frame['euler_3_proc'] = center_circ_array(data_frame['euler_3'].to_numpy())

    # normalize the data
    for collum in TDWDataset.norm_collums:
        normed_data_frame[collum] = (data_frame[collum] - data_frame[collum].mean()) / data_frame[collum].std()
        normed_data_frame[collum] = normed_data_frame[collum].astype(np.float32)
    
    # save the processed data
    normed_data_frame.to_csv(str(data_path.joinpath('images_meta_shuffled_processed.csv').resolve()))
    yaml_file_path = os.path.join(root_dir, 'mappings.yml')
    with open(yaml_file_path, 'w') as file:
        yaml.dump(mappings, file, default_flow_style=False)


if __name__ == '__main__':
    # run this script to save model scores for all models in an experiment
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', help='Name of the dataset')
    args = parser.parse_args()
    
    tdw_dataset_preprocess(root_dir = os.path.join('./data/', args.name))
