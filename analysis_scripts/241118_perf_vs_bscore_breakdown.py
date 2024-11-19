# %%
from pathlib import Path
import pandas as pd
import numpy as np

from config_global import EXP_DIR, DATA_DIR, ROOT_DIR
from utils import get_model_id
from exp_config_list import multi_task_resnet50_tdw_1m20240206_0908, multi_task_resnet50_tdw_1m20240206_earlier_0908

from scipy import stats
from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt
import easyfigs.basicplot as bp

# %%
config_list = multi_task_resnet50_tdw_1m20240206_0908() + multi_task_resnet50_tdw_1m20240206_earlier_0908()
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
task_list = ['distance_reg', 'translation_reg', 'rotation_reg', 'category_class', 'object_class']
region_list = ['V1', 'V2', 'V4', 'IT', 'Behavior']

task2loss_name = {
    'distance_reg': 'val_distance_reg_loss',
    'translation_reg': 'val_translation_reg_loss',
    'rotation_reg': 'val_rotation_reg_tdw_two_units_sin_cos_mse_loss',
    'category_class': 'val_category_class_loss',
    'object_class': 'val_object_class_loss',
}

task2loss_str = {
    'distance_reg': 'distance regression loss',
    'translation_reg': 'translation regression loss',
    'rotation_reg': 'rotation regression loss',
    'category_class': 'category classification loss',
    'object_class': 'identity classification loss',
}

task2tgt = {
    'distance_reg': 'Distance',
    'translation_reg': 'Translation',
    'rotation_reg': 'Rotation',
    'category_class': 'Obj. category',
    'object_class': 'Obj. identity',
}

def perf_vs_alginment(task, region):
    data = df[(df['exp_group'] == task) & (df['benchmark_region'] == region)]
    x = - np.array(data[task2loss_name[task]])
    y = np.array(data['score'])
    batch_num = np.array(data['batch'])
    r, p_val = stats.pearsonr(x, y)

    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.scatter(x, y, c=batch_num, alpha=0.8, cmap='cool', norm='log')
    pv_str = f'p-value = {p_val:.1e}' if p_val > 1e-3 else 'p-value < 1e-10'
    ax.text(0.05, 0.8, f'Pearson r = {r:.3f}\n{pv_str}', transform=ax.transAxes)
    ax.set_xlabel(f'Negative {task2loss_str[task]}')
    ax.set_ylabel(f'{region} alignment score')
    ax.set_title(f'Model target: {task2tgt[task]}')
    ax.set_xticks([-1.0, -0.5, 0])
    # ax.set_yticks([0.3, 0.4])
    bp.remove_top_right_spines(ax)
    fig.savefig(Path(ROOT_DIR).joinpath(f'figures/{task2loss_name[task]}_vs_{region}_score.pdf'), transparent=True, bbox_inches='tight')

# %%
for region in region_list:
    perf_vs_alginment('distance_reg', region)

# %%
for task in task_list:
    for region in region_list:
        perf_vs_alginment(task, region)

# %%
data = df_neural[df_neural['exp_group'] == 'distance_reg'].groupby('model')[['score', 'val_distance_reg_loss', 'batch']].mean()
x = - np.array(data['val_distance_reg_loss'])
y = np.array(data['score'])
batch_num = np.array(data['batch'])
r, p_val = stats.pearsonr(x, y)
print(f'Distance regression: Pearson r = {r:.3f}, p-value = {p_val:.1e}')

data = df_neural[df_neural['exp_group'] == 'translation_reg'].groupby('model')[['score', 'val_translation_reg_loss', 'batch']].mean()
x = - np.array(data['val_translation_reg_loss'])
y = np.array(data['score'])
batch_num = np.array(data['batch'])
r, p_val = stats.pearsonr(x, y)
print(f'Translation regression: Pearson r = {r:.3f}, p-value = {p_val:.1e}')

data = df_neural[df_neural['exp_group'] == 'rotation_reg'].groupby('model')[['score', 'val_rotation_reg_tdw_two_units_sin_cos_mse_loss', 'batch']].mean()
x = - np.array(data['val_rotation_reg_tdw_two_units_sin_cos_mse_loss'])
y = np.array(data['score'])
batch_num = np.array(data['batch'])
r, p_val = stats.pearsonr(x, y)
print(f'Rotation regression: Pearson r = {r:.3f}, p-value = {p_val:.1e}')

data = df_neural[df_neural['exp_group'] == 'category_class'].groupby('model')[['score', 'val_category_class_loss', 'batch']].mean()
x = - np.array(data['val_category_class_loss'])
y = np.array(data['score'])
batch_num = np.array(data['batch'])
r, p_val = stats.pearsonr(x, y)
print(f'Object category classification: Pearson r = {r:.3f}, p-value = {p_val:.1e}')

data = df_neural[df_neural['exp_group'] == 'object_class'].groupby('model')[['score', 'val_object_class_loss', 'batch']].mean()
x = - np.array(data['val_object_class_loss'])
y = np.array(data['score'])
batch_num = np.array(data['batch'])
r, p_val = stats.pearsonr(x, y)
print(f'Object identity classification: Pearson r = {r:.3f}, p-value = {p_val:.1e}')


# %%



