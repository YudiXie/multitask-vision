import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import circmean
from PIL import Image
import torch
from torch.utils.data import Dataset
import yaml


def get_image_meta_path(index_df, idx, dataset_path):
    """
    Get the image path and meta data path for a given index.
    """
    image_idx, scene_name, wnid, model_name = index_df.iloc[idx]

    record_path = dataset_path.joinpath('images', scene_name, wnid, model_name)
    img_path = record_path.joinpath(f"img_img_{image_idx:010d}.jpg")
    img_meta_path = record_path.joinpath(f"img_{image_idx:010d}_info.csv")
    return img_path, img_meta_path


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

    # all columns in the stimulus set
    all_columns = ['id', 'background_id', 's', 'image_id', 'image_file_name', 'filename', 'rxy', 'tz', 'category_name', 'rxz_semantic', 'ty', 'ryz', 'object_name', 'variation', 'size', 'rxy_semantic', 'ryz_semantic', 'rxz']
    # columns that need to be normalized
    norm_columns = ['s', 'ty', 'tz', 'rxy', 'rxz', 'ryz', 'rxy_semantic', 'rxz_semantic', 'ryz_semantic']

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
        for column in self.norm_columns:
            normed_data_frame[column] = (data_frame[column] - data_frame[column].mean()) / data_frame[column].std()
            normed_data_frame[column] = normed_data_frame[column].astype(np.float32)
        
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

        for column in self.norm_columns:
            sample[column] = self.normed_data_frame.loc[idx, column]

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
    norm_columns = ['neg_x', 'ty', 'tz'] # columns that need to be normalized

    def __init__(self,
                 root_dir='./data/tdw_image_dataset_small_multi_env_hdri',
                 split='train',
                 transform=None,
                 fraction=1.0,
                 ):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            fraction (float): fraction of the training or testing dataset to use, 1.0 means use all the data
                this is used to investigate the scaling of the training with the dataset size
        """
        self.root_path = Path(root_dir)
        self.transform = transform

        with open(self.root_path.joinpath('mappings.yml'), 'r') as file:
            mappings = yaml.safe_load(file)
        self.mappings = mappings

        self.headers = self.root_path.joinpath('img_meta_headers.txt').read_text(encoding="utf-8").split("\n")
        self.means_stds = pd.read_csv(self.root_path.joinpath('norm_column_mean_std.csv'), index_col=0).iloc[0]
        
        dataset_index = pd.read_csv(self.root_path.joinpath('index_img_shuffled.csv'), index_col=0)
        full_dset_size = len(dataset_index)
        split_index = round(full_dset_size * 0.8) if full_dset_size * 0.2 < 50000 else -50000

        if split == 'all':
            pass
        elif split == 'train':
            dataset_index = dataset_index[:split_index]
        elif split == 'val':
            dataset_index = dataset_index[split_index:].reset_index(drop=True)
        else:
            raise ValueError('split must be either all, train, or val')
        
        self.dataset_index = dataset_index[:round(len(dataset_index) * fraction)]

    def __len__(self):
        return len(self.dataset_index)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path, img_meta_path = get_image_meta_path(self.dataset_index, idx, self.root_path)

        image = load_image(img_path)
        if self.transform:
            image = self.transform(image)
        sample = {'image': image}
        sample['category_label'] = self.mappings['category_str2int'][self.dataset_index.iloc[idx]['wnid']]
        sample['object_label'] = self.mappings['object_str2int'][self.dataset_index.iloc[idx]['model']]
        
        for i in range(2, 9):
            # reduce the 8 category labels (0..7) to 2, 3, 4, 5, 6, 7, 8 category labels
            sample[f'cat_label_reduce{i}'] = sample['category_label'] if sample['category_label'] < i else (i - 1)

        img_meta = pd.read_csv(img_meta_path, names=self.headers).iloc[0]
        for i in range(1, 4):
            sample[f'euler_{i}'] = np.float32(img_meta[f'euler_{i}'])
        
        for column in self.norm_columns:
            sample[column] = np.float32((img_meta[column] - self.means_stds[f'{column}_mean']) / self.means_stds[f'{column}_std'])

        return sample
