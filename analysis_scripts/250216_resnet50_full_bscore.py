# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import easyfigs.basicplot as bp

# %%
df = pd.read_csv('experiments/multi_task_resnet50_tdw_1m20240206_0908/250215_full_brainscore_benchmark_scores.csv', index_col=0)

name2outputunits = {
    'yudixie_resnet50_random_0_240908': 0,
    'yudixie_resnet50_distance_reg_0_240908': 1,
    'yudixie_resnet50_translation_reg_0_240908': 2,
    'yudixie_resnet50_distance_translation_0_240908': 3,
    'yudixie_resnet50_rotation_reg_0_240908': 6,
    'yudixie_resnet50_distance_rotation_0_240908': 7,
    'yudixie_resnet50_translation_rotation_0_240908': 8,
    'yudixie_resnet50_distance_translation_rotation_0_240908': 9,
    'yudixie_resnet50_category_class_0_240908': 117,
    'yudixie_resnet50_object_class_0_240908': 548,
    'yudixie_resnet50_cat_obj_class_all_latents_0_240908': 674,
    'yudixie_resnet50_imagenet1kpret_0_240908': 1000,
}
model_name_list = list(name2outputunits.keys())
output_units = list(name2outputunits.values())

df_reindex = df.reindex(model_name_list)

# %%
benchmarks_dict = {
    'V1': ['FreemanZiemba2013.V1-pls', 'Marques2020'],
    'V2': ['FreemanZiemba2013.V2-pls', ],
    'V4': ['MajajHong2015.V4-pls', 'SanghaviMurty2020.V4-pls', 'Sanghavi2020.V4-pls', 'SanghaviJozwik2020.V4-pls'],
    'IT': ['MajajHong2015.IT-pls', 'SanghaviMurty2020.IT-pls', 'Sanghavi2020.IT-pls', 'SanghaviJozwik2020.IT-pls', 'Bracci2019.anteriorVTC-rdm'],
    'Behavior': ['Rajalingham2018-i2n', 'Maniquet2024', 'Ferguson2024', 'Hebart2023-match', 'tong.Coggan2024_behavior-ConditionWiseAccuracySimilarity'],
}

# %%
def plot_bscore(region_name, bcm_name, results):
    fig, ax = plt.subplots(figsize=(4.8, 3.6))

    ax.axhline(results[0], color='k', linestyle=':', label='Untrained', alpha=0.5)
    ax.scatter(output_units[1:8], results[1:8], label='Spatial latents (TDW-117)', c='#448aff')
    ax.scatter(output_units[8], results[8], label='Object category (TDW-117)', c='#ff9800')
    ax.scatter(output_units[9], results[9], label='Object identity (TDW-117)', c='#DA814E')
    ax.scatter(output_units[10], results[10], label='All spatial + classification (TDW-117)', c='#8bc34a')
    ax.scatter(output_units[11], results[11], label='Object category (ImageNet-1K)', marker='D', c='#f44336')
    
    ax.set_xscale('log')
    ax.set_ylim(0, 1)

    ax.legend(fontsize=8, title='Training targets')
    ax.set_xlabel('Number of supervised output units')
    ax.set_ylabel(f'{bcm_name} score')
    ax.set_title(bcm_name)
    bp.remove_top_right_spines(ax)
    fig.savefig(f'figures/full_bscore_{region_name}_{bcm_name}.pdf', transparent=True, bbox_inches='tight')

# %%
full_score = []
full_nerual = []
for region, benchmark_list in benchmarks_dict.items():
    region_score = []
    for col in benchmark_list:
        bchm_res = df_reindex[col].tolist()
        assert not any(pd.isna(bchm_res))

        plot_bscore(region, col, bchm_res)
        region_score.append(bchm_res)
    
    region_score_mean = np.mean(np.array(region_score), axis=0)
    plot_bscore(region, f'{region} mean', region_score_mean)

    full_score.append(region_score_mean)
    if region in ['V1', 'V2', 'V4', 'IT']:
        full_nerual.append(region_score_mean)

full_score_mean = np.mean(np.array(full_score), axis=0)
plot_bscore('all', 'All benchmarks mean', full_score_mean)

full_nerual_mean = np.mean(np.array(full_nerual), axis=0)
plot_bscore('all', 'Neural benchmarks mean', full_nerual_mean)

# %%



