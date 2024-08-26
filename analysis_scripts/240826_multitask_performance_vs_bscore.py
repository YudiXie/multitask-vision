# %%
from pathlib import Path
import pandas as pd
import numpy as np

from config_global import EXP_DIR, DATA_DIR, ROOT_DIR
from utils import get_model_id
from exp_config_list import multi_task_resnet50_tdw_10m20240208_0802, multi_task_resnet50_tdw_10m20240208_earlier_0817

from scipy import stats
from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt
import easyfigs.basicplot as bp

# %%
config_list = multi_task_resnet50_tdw_10m20240208_0802() + multi_task_resnet50_tdw_10m20240208_earlier_0817()
all_tasks = ['distance_reg', 'translation_reg', 'rotation_reg_tdw_two_units_sin_cos_mse', 'object_class', 'category_class']
save_dict = {'model': [],
             'batch': [],
             'benchmark_region': [],
             'score': [],
             'exp_group': [],
             }
for task in all_tasks:
    save_dict[f'val_{task}_loss'] = []
save_dict['val_category_acc'] = []
save_dict['val_object_acc'] = []

benchmark_dict = {
    'V1': 'FreemanZiemba2013public.V1-pls',
    'V2': 'FreemanZiemba2013public.V2-pls',
    'V4': 'MajajHong2015public.V4-pls',
    'IT': 'MajajHong2015public.IT-pls',
    'Behavior': 'Rajalingham2018public-i2n',
    }

for config in config_list:    
    for batch_n in config['score_model_nums']:
        model_id = get_model_id(config) + f'-batch-{batch_n}'
        val_results = pd.read_csv(Path(config['save_path']).joinpath(f'val_results_batch_n_{batch_n}.csv'), index_col=0)

        for region, benchmark_id in benchmark_dict.items():
            score_path = Path(DATA_DIR).joinpath(f'{model_id}_{benchmark_id}_score.csv')
            score = pd.read_csv(score_path, index_col=0)['score'][0]

            save_dict['model'].append(model_id)
            save_dict['batch'].append(batch_n)
            save_dict['benchmark_region'].append(region)
            save_dict['score'].append(score)
            save_dict['exp_group'].append(config['group_name'])

            train_tasks = config['tasks']
            no_results_tasks = [task for task in all_tasks if task not in train_tasks]
            for task in train_tasks:
                save_dict[f'val_{task}_loss'].append(val_results.loc[f'val_{task}_loss', '0'])
                if task == 'object_class':
                    save_dict['val_object_acc'].append(val_results.loc['val_object_acc', '0'])
                if task == 'category_class':
                    save_dict['val_category_acc'].append(val_results.loc['val_category_acc', '0'])
            for task in no_results_tasks:
                save_dict[f'val_{task}_loss'].append(np.nan)
                if task == 'object_class':
                    save_dict['val_object_acc'].append(np.nan)
                if task == 'category_class':
                    save_dict['val_category_acc'].append(np.nan)

df = pd.DataFrame.from_dict(save_dict)
df_neural = df[df['benchmark_region'] != 'Behavior']

# %%
data = df_neural[df_neural['exp_group'] == 'distance_reg'].groupby('model')[['score', 'val_distance_reg_loss', 'batch']].mean()
# 2 outliers out of 104 data points where the loss is lareger than 1.0 is excluded
data = data[data['val_distance_reg_loss'] < 1.0]

x = - np.array(data['val_distance_reg_loss'])
y = np.array(data['score'])
batch_num = np.array(data['batch'])
r, p_val = stats.pearsonr(x, y)

fig, ax = plt.subplots(figsize=(3.6, 2.7))
ax.scatter(x, y, c=batch_num, alpha=0.8, cmap='cool', norm='log')
ax.text(0.5, 0.05, f'Pearson r = {r:.2f}\np-value = {p_val:.1e}', transform=ax.transAxes)
ax.set_xlabel('Negative distance regression loss')
ax.set_ylabel('Mean neural alignment')
ax.set_xticks([-1, -0.5, 0])
ax.set_yticks([0.3, 0.4])
bp.remove_top_right_spines(ax)
fig.tight_layout()
fig.savefig(Path(ROOT_DIR).joinpath(f'figures/distance_reg_loss_vs_score.pdf'), transparent=True)

fig, ax = plt.subplots(figsize=(3.6, 2.7))
sc = ax.scatter(x, y, c=batch_num, alpha=0.8, cmap='cool', norm='log')
cbar = fig.colorbar(sc, label='Num. of training batches')
fig.tight_layout()
fig.savefig(Path(ROOT_DIR).joinpath(f'figures/distance_reg_loss_vs_score_colorbar.pdf'), transparent=True)

# %%
data = df_neural[df_neural['exp_group'] == 'translation_reg'].groupby('model')[['score', 'val_translation_reg_loss', 'batch']].mean()

x = - np.array(data['val_translation_reg_loss'])
y = np.array(data['score'])
batch_num = np.array(data['batch'])
r, p_val = stats.pearsonr(x, y)

fig, ax = plt.subplots(figsize=(3.6, 2.7))
ax.scatter(x, y, c=batch_num, alpha=0.8, cmap='cool', norm='log')
ax.text(0.5, 0.05, f'Pearson r = {r:.2f}\np-value = {p_val:.1e}', transform=ax.transAxes)
ax.set_xlabel('Negative translation regression loss')
ax.set_ylabel('Mean neural alignment')
ax.set_xticks([-0.5, 0])
ax.set_yticks([0.3, 0.4])
bp.remove_top_right_spines(ax)
fig.tight_layout()
fig.savefig(Path(ROOT_DIR).joinpath(f'figures/translation_reg_loss_vs_score.pdf'), transparent=True)

# %%
data = df_neural[df_neural['exp_group'] == 'rotation_reg'].groupby('model')[['score', 'val_rotation_reg_tdw_two_units_sin_cos_mse_loss', 'batch']].mean()

x = - np.array(data['val_rotation_reg_tdw_two_units_sin_cos_mse_loss'])
y = np.array(data['score'])
batch_num = np.array(data['batch'])
r, p_val = stats.pearsonr(x, y)

fig, ax = plt.subplots(figsize=(3.6, 2.7))
ax.scatter(x, y, c=batch_num, alpha=0.8, cmap='cool', norm='log')
ax.text(0.5, 0.05, f'Pearson r = {r:.2f}\np-value = {p_val:.1e}', transform=ax.transAxes)
ax.set_xlabel('Negative rotation regression loss')
ax.set_ylabel('Mean neural alignment')
ax.set_xticks([-0.1, 0])
ax.set_yticks([0.2, 0.3, 0.4])
bp.remove_top_right_spines(ax)
fig.tight_layout()
fig.savefig(Path(ROOT_DIR).joinpath(f'figures/rotation_reg_loss_vs_score.pdf'), transparent=True)

# %%
data = df_neural[df_neural['exp_group'] == 'category_class'].groupby('model')[['score', 'val_category_acc', 'batch']].mean()

x = np.array(data['val_category_acc'])
y = np.array(data['score'])
batch_num = np.array(data['batch'])
r, p_val = stats.pearsonr(x, y)

fig, ax = plt.subplots(figsize=(3.6, 2.7))
ax.scatter(x, y, c=batch_num, alpha=0.8, cmap='cool', norm='log')
ax.text(0.5, 0.05, f'Pearson r = {r:.2f}\np-value = {p_val:.1e}', transform=ax.transAxes)
ax.set_xlabel('Object category classification accuracy')
ax.set_ylabel('Mean neural alignment')
ax.set_xticks([0.0, 0.5, 1.0])
ax.set_yticks([0.2, 0.3, 0.4])
bp.remove_top_right_spines(ax)
fig.tight_layout()
fig.savefig(Path(ROOT_DIR).joinpath(f'figures/obj_cat_acc_vs_score.pdf'), transparent=True)

# %%
data = df_neural[df_neural['exp_group'] == 'object_class'].groupby('model')[['score', 'val_object_acc', 'batch']].mean()

x = np.array(data['val_object_acc'])
y = np.array(data['score'])
batch_num = np.array(data['batch'])
r, p_val = stats.pearsonr(x, y)

fig, ax = plt.subplots(figsize=(3.6, 2.7))
ax.scatter(x, y, c=batch_num, alpha=0.8, cmap='cool', norm='log')
ax.text(0.5, 0.05, f'Pearson r = {r:.2f}\np-value = {p_val:.1e}', transform=ax.transAxes)
ax.set_xlabel('Object identity classification accuracy')
ax.set_ylabel('Mean neural alignment')
ax.set_xticks([0.0, 0.5, 1.0])
ax.set_yticks([0.2, 0.3, 0.4])
bp.remove_top_right_spines(ax)
fig.tight_layout()
fig.savefig(Path(ROOT_DIR).joinpath(f'figures/obj_id_acc_vs_score.pdf'), transparent=True)


