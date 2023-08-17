
import os
from pathlib import Path

from skimage import io

import numpy as np
import pandas as pd
from scipy.stats import circmean
from PIL import Image

import torch
from torch.utils.data import Dataset

RNG: np.random.RandomState = np.random.RandomState(0)


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
        self.category_str2int = {}
        cat_list = list(normed_data_frame['category_name'].unique())
        cat_list.sort()
        for i, category_name in enumerate(cat_list):
            self.category_str2int[category_name] = i
        self.category_int2str = {v: k for k, v in self.category_str2int.items()}
        normed_data_frame['category_label'] = [self.category_str2int[cn] for cn in normed_data_frame['category_name']]
        
        # create a map from object name to object label
        self.object_str2int = {}
        object_list = list(normed_data_frame['object_name'].unique())
        object_list.sort()
        for i, object_name in enumerate(object_list):
            self.object_str2int[object_name] = i
        self.object_int2str = {v: k for k, v in self.object_str2int.items()}
        normed_data_frame['object_label'] = [self.object_str2int[on] for on in normed_data_frame['object_name']]

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
    """
    norm_collums = ['neg_x', 'ty', 'tz','euler_1_proc', 'euler_2_proc', 'euler_3_proc']

    def __init__(self, 
                 csv_file='./data/tdw_image_dataset_small/images_meta.csv',
                 root_dir='./data/tdw_image_dataset_small',
                 split='train',
                 transform=None,
                 ):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        data_frame = pd.read_csv(csv_file, index_col=0)
        data_frame = data_frame.sample(frac=1, random_state=RNG).reset_index(drop=True) # shuffle the data

        # check if all images exist
        data_path = Path(root_dir)
        for i in range(len(data_frame)):
            assert data_path.joinpath(data_frame.loc[i, 'image_filename']).is_file()

        normed_data_frame = data_frame.copy()
        self.root_dir = root_dir
        self.transform = transform

        # process the euler angles
        data_frame['euler_1_proc'] = center_circ_array(data_frame['euler_1'].to_numpy())
        data_frame['euler_2_proc'] = center_circ_array(data_frame['euler_2'].to_numpy())
        data_frame['euler_3_proc'] = center_circ_array(data_frame['euler_3'].to_numpy())
      
        # create a map from category name to category label
        self.category_str2int = {}
        cat_list = list(normed_data_frame['wnid'].unique())
        cat_list.sort()
        for i, category_name in enumerate(cat_list):
            self.category_str2int[category_name] = i
        self.category_int2str = {v: k for k, v in self.category_str2int.items()}
        normed_data_frame['category_label'] = [self.category_str2int[cn] for cn in normed_data_frame['wnid']]
        
        # create a map from object name to object label
        self.object_str2int = {}
        object_list = list(normed_data_frame['record_name'].unique())
        object_list.sort()
        for i, object_name in enumerate(object_list):
            self.object_str2int[object_name] = i
        self.object_int2str = {v: k for k, v in self.object_str2int.items()}
        normed_data_frame['object_label'] = [self.object_str2int[on] for on in normed_data_frame['record_name']]

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


if __name__ == '__main__':
    pass
