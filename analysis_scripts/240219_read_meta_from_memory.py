# %%
from pathlib import Path
import pandas as pd
from tqdm import trange
from dataset import get_image_meta_path
from dataset import TDWDataset
from collections import defaultdict
import numpy as np

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
# create data to append to the dataframe
# read CSV files string convertion
# reverse indexing to avoid cases like "bag, handbag, pocketbook, purse
inv_idx = {header: - (len(meta_headers) - i) for i, header in enumerate(meta_headers)}
append_data = defaultdict(list)
for i_row in trange(len(index_df)):
    _, img_meta_path = get_image_meta_path(index_df, i_row, dataset_path)
    data_list = img_meta_path.read_text().split(",")
    for column in TDWDataset.vis_collumns:
        append_data[column].append(float(data_list[inv_idx[column]]))

# %%
for k, v in append_data.items():
    index_df[k] = np.array(v, dtype=np.float32)
index_df.to_csv(dataset_path.joinpath('index_img_shuffled_with_meta.csv'))

# %%
# read meta data from memory directly
x_sum = 0
for i_row in trange(sample_size):
    x_sum += index_df.iloc[i_row]['rel_pos_z']
x_sum

# %%
# check if the new index is correct
# reverse indexing to avoid cases like "bag, handbag, pocketbook, purse"
sample_i = np.random.randint(len(index_df))
_, img_meta_path = get_image_meta_path(index_df, sample_i, dataset_path)
print(meta_headers)
print(img_meta_path.read_text())
print(index_df.iloc[sample_i])

# %%



