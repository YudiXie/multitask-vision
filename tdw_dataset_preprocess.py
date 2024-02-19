from pathlib import Path
import argparse
from collections import defaultdict

import numpy as np
from tqdm import trange
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from dataset import TDWDataset, create_mapping, get_image_meta_path
from plots import data_hist


if __name__ == '__main__':
    """
    Preprocess the TDW dataset, including:
    1. shuffle the index
    2. create a map from category name to category label
    3. create a map from object name to object label
    4. calculate and stores the mean and std of columns in norm_columns by sampling the dataset
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default='', help='the path to saved index')
    # eg. '/om2/user/yu_xie/data/tdw_images/tdw_image_dataset_small_multi_env_hdri/index_img_5898.csv'
    args = parser.parse_args()

    index_path = Path(args.index)
    dataset_path = index_path.parent
    # check if all images exist
    assert dataset_path.joinpath("dataset_complete.txt").is_file(), "No complete check file found"

    index_df = pd.read_csv(index_path, names=['image_index', 'scene', 'wnid', 'model'], skiprows=1)
    meta_headers = dataset_path.joinpath('img_meta_headers.txt').read_text(encoding="utf-8").split("\n")
    # shuffle the index, and save a copy on disk
    shuffled_index = index_df.sample(frac=1).reset_index(drop=True)
    shuffled_index.to_csv(dataset_path.joinpath("index_img_shuffled.csv"))

    # concatenate the meta data to the index and save a copy on disk
    # read CSV files string convertion, reverse indexing to avoid cases like "bag, handbag, pocketbook, purse
    inv_idx = {header: - (len(meta_headers) - i) for i, header in enumerate(meta_headers)}
    append_data = defaultdict(list)
    for i_row in trange(len(shuffled_index)):
        _, img_meta_path = get_image_meta_path(shuffled_index, i_row, dataset_path)
        data_list = img_meta_path.read_text().split(",")
        for column in TDWDataset.vis_collumns:
            append_data[column].append(float(data_list[inv_idx[column]]))
    
    for k, v in append_data.items():
        shuffled_index[k] = np.array(v, dtype=np.float32)
    
    shuffled_index.to_csv(dataset_path.joinpath('index_img_shuffled_with_meta.csv'))

    # --------------------------------
    # create a map from category name to category label
    category_str2int, category_int2str = create_mapping(list(shuffled_index['wnid'].unique()))
    # create a map from object name to object label
    object_str2int, object_int2str = create_mapping(list(shuffled_index['model'].unique()))

    mappings = {
        'category_str2int': category_str2int,
        'category_int2str': category_int2str,
        'object_str2int': object_str2int,
        'object_int2str': object_int2str,
    }

    with open(dataset_path.joinpath('mappings.yml'), 'w') as yamlfile:
        yaml.dump(mappings, yamlfile, default_flow_style=False)

    # --------------------------------
    # sample some data to calculate the mean and std of columns in norm_columns
    sample_size = min(100000, len(shuffled_index))
    sample_index = shuffled_index.iloc[:sample_size]
    
    img_meta_rows = []
    for i_row in trange(len(sample_index)):
        _, img_meta_path = get_image_meta_path(sample_index, i_row, dataset_path)
        img_meta_rows.append(pd.read_csv(img_meta_path, names=meta_headers))
    img_meta_df = pd.concat(img_meta_rows, ignore_index=True)

    # calculate the mean and std of columns in norm_columns
    mean_std_dict = {}
    for column in TDWDataset.norm_columns:
        mean_std_dict[f'{column}_mean'] = [img_meta_df[column].mean(), ]
        mean_std_dict[f'{column}_std'] = [img_meta_df[column].std(), ]
    
    # visulize the target distribution
    for column in TDWDataset.vis_collumns:
        show_mean_std = True if column in TDWDataset.norm_columns else False
        fig, ax = plt.subplots()
        data_hist(ax, np.array(img_meta_df[column]), 
                  xlabel=column, show_mean_std=show_mean_std)
        fig.savefig(dataset_path.joinpath(f'img_meta_hist_{column}.pdf'), transparent=True)
    
    # save the mean and std to be used later
    pd.DataFrame.from_dict(mean_std_dict).to_csv(dataset_path.joinpath('norm_column_mean_std.csv'))
