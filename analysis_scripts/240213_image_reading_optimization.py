# %%
from pathlib import Path
import pandas as pd
from tqdm import trange
from dataset import get_image_meta_path
import numpy as np
import csv
from dataset import load_image
from PIL import Image
from torchvision.datasets.folder import pil_loader, default_loader

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
    img_path, _ = get_image_meta_path(index_df, i_row, dataset_path)
    x_sum += 0.1
x_sum

# %%
# use previous image loading
x_sum = 0
for i_row in trange(sample_size):
    img_path, _ = get_image_meta_path(index_df, i_row, dataset_path)
    img = load_image(img_path)
    x_sum += 0.1
x_sum

# %%
# use PIL directly without copying, this is a lazy operation
x_sum = 0
for i_row in trange(sample_size):
    img_path, _ = get_image_meta_path(index_df, i_row, dataset_path)
    img = Image.open(img_path)
    x_sum += 0.1
x_sum

# %%
# use PIL loader from torchvision
x_sum = 0
for i_row in trange(sample_size):
    img_path, _ = get_image_meta_path(index_df, i_row, dataset_path)
    img = pil_loader(img_path)
    x_sum += 0.1
x_sum

# %%
# use default_loader loader from torchvision
x_sum = 0
for i_row in trange(sample_size):
    img_path, _ = get_image_meta_path(index_df, i_row, dataset_path)
    img = default_loader(img_path)
    x_sum += 0.1
x_sum


