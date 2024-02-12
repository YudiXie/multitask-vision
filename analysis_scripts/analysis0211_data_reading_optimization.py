# %%
from pathlib import Path
import pandas as pd
from tqdm import trange
from dataset import get_image_meta_path
import numpy as np
import csv

# %%
index_path = Path('/om/user/yu_xie/data/tdw_images/tdw_image_dataset_1m_20240206/index_img_shuffled.csv')
dataset_path = index_path.parent
index_df = pd.read_csv(index_path, index_col=0)
meta_headers = dataset_path.joinpath('img_meta_headers.txt').read_text(encoding="utf-8").split("\n")

# %%
sample_size = 10000

# %%
# nothing
x_sum = 0
for i_row in trange(sample_size):
    _, img_meta_path = get_image_meta_path(index_df, i_row, dataset_path)
    x_sum += 0.1
x_sum

# %%
# read CSV files using pandas
x_sum = 0
for i_row in trange(sample_size):
    _, img_meta_path = get_image_meta_path(index_df, i_row, dataset_path)
    img_meta = pd.read_csv(img_meta_path, names=meta_headers).iloc[0]
    x_sum += np.float32(img_meta[f'rel_pos_z'])
x_sum


# %%
# read CSV files string convertion
x_sum = 0
# reverse indexing to avoid cases like "bag, handbag, pocketbook, purse"
list_idx = - (len(meta_headers) - meta_headers.index('rel_pos_z'))
for i_row in trange(sample_size):
    _, img_meta_path = get_image_meta_path(index_df, i_row, dataset_path)
    read_x = float(img_meta_path.read_text().split(",")[list_idx])
    x_sum += read_x
x_sum

# %%
# read CSV files using csv module
x_sum = 0
# reverse indexing to avoid cases like "bag, handbag, pocketbook, purse"
list_idx = - (len(meta_headers) - meta_headers.index('rel_pos_z'))
for i_row in trange(sample_size):
    _, img_meta_path = get_image_meta_path(index_df, i_row, dataset_path)
    with open(img_meta_path) as csvfile:
        spamreader = csv.reader(csvfile)
        read_data = next(spamreader)
    x_sum += float(read_data[8])
x_sum

# %%
# read using numpy
for i_row in trange(sample_size):
    _, img_meta_path = get_image_meta_path(index_df, i_row, dataset_path)
    img_meta = pd.read_csv(img_meta_path, names=meta_headers).iloc[0]
    npy_path = img_meta_path.parent / (img_meta_path.name[:-4] + ".npy")
    np.save(npy_path, img_meta[6: 16].to_numpy(dtype=np.float32))

# %%
x_sum = 0
for i_row in trange(sample_size):
    _, img_meta_path = get_image_meta_path(index_df, i_row, dataset_path)
    npy_path = img_meta_path.parent / (img_meta_path.name[:-4] + ".npy")
    x_sum += np.load(npy_path)[2]
x_sum

# %%



