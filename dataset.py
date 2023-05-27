
import os

from skimage import io

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


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
    # my guess about the meaning of the collums
    # s (size)
    # tz, vertical translation
    # ty, horizontal translation
    # the rest seems to be rotations
    # it seems that the following is true most of the time, but not always
    # rxy_semantic - rxy = 90
    # rxz_semantic = rxz
    # ryz_semantic = ryz

    # all collums in the stimulus set
    all_collums = ['id', 'background_id', 's', 'image_id', 'image_file_name', 'filename', 'rxy', 'tz', 'category_name', 'rxz_semantic', 'ty', 'ryz', 'object_name', 'variation', 'size', 'rxy_semantic', 'ryz_semantic', 'rxz']
    # collums that need to be predicted
    pred_collums = ['category_name', 'object_name', 's', 'ty', 'tz', 'rxy', 'rxz', 'ryz']
    # collums that need to be normalized
    norm_collums = ['s', 'ty', 'tz', 'rxy', 'rxz', 'ryz']

    def __init__(self, 
                 csv_file='./data/image_dicarlo_hvm-public.csv', 
                 root_dir='./data/image_dicarlo_hvm-public', 
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
        cat_list = list(data_frame['category_name'].unique())
        cat_list.sort()
        for i, category_name in enumerate(cat_list):
            self.category_str2int[category_name] = i
        self.category_int2str = {v: k for k, v in self.category_str2int.items()}
        
        # create a map from object name to object label
        self.object_str2int = {}
        object_list = list(data_frame['object_name'].unique())
        object_list.sort()
        for i, object_name in enumerate(object_list):
            self.object_str2int[object_name] = i
        self.object_int2str = {v: k for k, v in self.object_str2int.items()}

        # normalize the data
        for collum in self.norm_collums:
            normed_data_frame[collum] = (data_frame[collum] - data_frame[collum].mean()) / data_frame[collum].std()
            normed_data_frame[collum] = normed_data_frame[collum].astype(np.float32)
        
        if split == 'train':
            self.normed_data_frame = normed_data_frame[:int(len(normed_data_frame) * 0.8)]
        elif split == 'val':
            self.normed_data_frame = normed_data_frame[int(len(normed_data_frame) * 0.8):].reset_index(drop=True)
        else:
            raise ValueError('split must be either train or val')

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
        sample['category_label'] = self.category_str2int[self.normed_data_frame.loc[idx, 'category_name']]
        sample['object_label'] = self.object_str2int[self.normed_data_frame.loc[idx, 'object_name']]

        for collum in self.norm_collums:
            sample[collum] = self.normed_data_frame.loc[idx, collum]

        return sample


if __name__ == '__main__':
    pass
