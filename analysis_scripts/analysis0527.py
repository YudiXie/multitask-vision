import os
from config_global import EXP_DIR
import pandas as pd
from exp_config_list import setup_config_list
from analysis import bar_2par, two_set_scatter_plot

df = pd.read_csv(os.path.join(EXP_DIR, 'multi_task_vs_categorization0527', 'mt0527_resnet18.csv'), index_col=0)

group_names = ['Pre-trained', 'Categorization', 'Multi-task']
x_axis_labels = ['V1', 'V2', 'V4', 'IT', 'Behavior']
benchmark_list = [
    'movshon.FreemanZiemba2013public.V1-pls',
    'movshon.FreemanZiemba2013public.V2-pls',
    'dicarlo.MajajHong2015public.V4-pls',
    'dicarlo.MajajHong2015public.IT-pls',
    'dicarlo.Rajalingham2018public-i2n',
    ]
data_dict = {}
error_dict = {}

for group in group_names:
    data_dict.update({group: list(df[df['exp_group'] == group].groupby('benchmark').mean()['score'].reindex(benchmark_list))})
    
for group in group_names[1:]:
    error_dict.update({group: list(df[df['exp_group'] == group].groupby('benchmark').std()['score'].reindex(benchmark_list))})

bar_2par(data_dict, x_axis_labels, group_names, error_dict,
        exp_name='multi_task_vs_categorization0527', 
        fig_name='compare_different_groups',
        y_label='Score',)

for i, (region, benchmark) in enumerate(zip(x_axis_labels, benchmark_list)):
    # each dot in the plot is an animal per sessions
    cat_data = df[(df['exp_group'] == 'Categorization') & (df['benchmark'] == benchmark)]['score']
    mul_data = df[(df['exp_group'] == 'Multi-task') & (df['benchmark'] == benchmark)]['score']
    two_set_scatter_plot(cat_data, mul_data,
                        labels=['Categorization', 'Multi-task'],
                        title_str=f'{region} Score',
                        ylabel='Score',
                        save_str=f'fig{i}_{region}_score_dot_random_seed')
