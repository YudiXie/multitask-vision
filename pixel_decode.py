import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from sklearn.random_projection import johnson_lindenstrauss_min_dim

from dataset import TDWDataset
from activity import cross_validate_on_target


# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
dataset = TDWDataset(root_dir='/om/user/yu_xie/data/tdw_images/tdw_1m_20240206',
                     split='val', fraction=0.04, transform=transform)

dset_index = dataset.dataset_index.copy()
cat_labels = [dataset.mappings['category_str2int'][wnid] for wnid in dset_index['wnid']]
dset_index['cat_labels'] = cat_labels

pixels = np.stack([np.array(dataset[i]['image']).reshape(-1) for i in range(len(dataset))])
targets = dset_index['cat_labels'].to_numpy(copy=True)
reduce_dim = johnson_lindenstrauss_min_dim(len(targets))

results = cross_validate_on_target(pixels, targets,
                                   downsample_method='random',
                                   downsample_number=reduce_dim,
                                   num_cross_val=10,
                                   mode='classification')

results_df = pd.DataFrame({'pixel_decoding': results})
results_df.to_csv('pixel_decoding_results.csv')
