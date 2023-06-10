import os
import sys
import subprocess
import copy
import argparse

from train import train_model
from config_global import ROOT_DIR, CONDA_ENV, CUDA_MODULE, EXP_DIR
from utils import save_config


def check_run_complete(cfg) -> bool:
    """
    Check if the run is complete by checking the complete file
    :param cfg: config
    :return: True if train is complete, False otherwise
    """

    exp_str = cfg['save_path'].split('/')[-1]
    complete_path = os.path.join(cfg['save_path'], 'train_complete.txt')
    
    if not os.path.exists(complete_path):
        print('No complete record for: ' + exp_str)
        run_complete = False
    else:
        run_complete = True
    
    return run_complete


def get_missing_runs(config_list: list) -> list:
    """
    Check if there are any missing runs in the experiments
    :param config_list: a list of experimental configs
    :return: mis_config_list, a list of missing runs
    """
    run_n_cmplt = [not check_run_complete(cfg) for cfg in config_list]
    if all([not val for val in run_n_cmplt]):
        print('All runs completed!')
    return [config_list[i] for i in range(len(config_list)) if run_n_cmplt[i]]


def get_jobfile(cmd,
                job_name,
                dep_ids=None,
                email='',
                sbatch_path='./sbatch/',
                output_path='./sbatch/',
                hours=8,
                partition=['normal'],
                cpu=4,
                mem=32,
                gpu_constraint='high-capacity',
                cuda_module='openmind8/cuda/11.7',
                conda_env='base',
                work_dir='./',
                ):
    """
    Create a job file.

    Args:
        cmd: python command to be execute by the cluster
        job_name: str, name of the job file
        dep_ids: None or a list of job ids used for job dependency
        email: str, email to send about job status
        sbatch_path : str, Directory to store the .sh file for sbatch
        output_path : str, Directory to store output of runs
        hours : int, number of hours to train
        partition : list, a list of cluster partition to use
        cpu : int, number of cpu cores to use
        mem : int, number of memory to use in GB
        gpu_constraint : str, gpu constraint to use
        cuda_module : str, cuda module to load
        conda_env : str, conda environment to use
        work_dir : str, working directory to execute the command
    Returns:
        job_file : str, Path to the job file.
    """
    if dep_ids is None:
        dep_ids = []
    assert type(dep_ids) is list, 'dependency ids must be list'
    assert all(type(id_) is str for id_ in dep_ids), 'dependency ids must all be strings'

    if len(dep_ids) == 0:
        dependency_line = ''
    else:
        dependency_line = '#SBATCH --dependency=afterok:' \
                          + ':'.join(dep_ids) + '\n'

    if email == '':
        email_line = ''
    else:
        email_line = '#SBATCH --mail-type=ALL\n' + \
                     '#SBATCH --mail-user={}\n'.format(email)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(sbatch_path):
        os.makedirs(sbatch_path)
    job_file = os.path.join(sbatch_path, job_name + '.sh')    
    with open(job_file, 'w') as f:
        f.write(
            '#!/bin/bash\n'
            + '#SBATCH -t {}:00:00\n'.format(hours)
            + '#SBATCH -N 1\n'
            + '#SBATCH -n {}\n'.format(cpu)
            + '#SBATCH --mem={}G\n'.format(mem)
            + '#SBATCH --gres=gpu:1\n'
            + '#SBATCH --constraint={}\n'.format(gpu_constraint)
            + '#SBATCH --partition={}\n'.format(','.join(partition))
            + '#SBATCH -e {}/slurm-%j-{}.out\n'.format(output_path, job_name)
            + '#SBATCH -o {}/slurm-%j-{}.out\n'.format(output_path, job_name)
            + dependency_line
            + email_line
            + '\n'
            + 'source ~/.bashrc\n'
            + 'module load {}\n'.format(cuda_module)
            + 'conda activate {}\n'.format(conda_env)
            + 'cd {}\n'.format(work_dir)
            + cmd + '\n'
            + '\n'
            )
    print(f'Created job file: {job_file}')
    return job_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cluster', action='store_true', help='Use batch submission on cluster')
    parser.add_argument('-p', '--partition', nargs='+', default=['normal'], help='Partition of resource on cluster to use')
    parser.add_argument('-m', '--missing', action='store_true', help='Run missing experiments')
    args = parser.parse_args()

    base_config = {
        'seed': 0,
        'run_id': 0,
        'batch_size': 64,
        'lr': 1e-3,
        'max_batch': 1000,
        'eval_per': 10,
        'group_name': 'multi_task',
        'tasks': [
            'category_class',
            'object_class',
            'rotation_reg',
            'size_reg',
            'translation_reg',
            ],
        'model_name': 'resnet18',
        'experiment_name': 'multi_task_0610',
        'save_path': './experiments/',
        }

    task_set_dict = {
        'multi_task': ['category_class', 'object_class', 'rotation_reg', 'size_reg', 'translation_reg'],
        'categorization': ['category_class'],
        'multi_task_wo_object_class': ['category_class', 'rotation_reg', 'size_reg', 'translation_reg'],
        'size_reg': ['size_reg'],
        'translation_reg': ['translation_reg'],
        'rotation_reg': ['rotation_reg'],
    }
    seed_list = [0, 1, 2, 3, 4]
    
    config_list = []
    run_id = 0
    for group_n, task_set in task_set_dict.items():
        for seed in seed_list:
            cfg = copy.deepcopy(base_config)
            cfg['group_name'] = group_n
            cfg['tasks'] = task_set
            cfg['seed'] = seed

            cfg['save_path'] = os.path.join(EXP_DIR, cfg['experiment_name'], f'run_{run_id:04d}')
            cfg['run_id'] = run_id
            config_list.append(cfg)
            run_id += 1
    
    if args.missing:
        config_list = get_missing_runs(config_list)
        if input('Continue to submit? (yes/no): ') != 'yes':
            sys.exit("exit program.")
    
    for config in config_list:
        config_file_path = save_config(config, config['save_path'])
        if not args.cluster:
            # run it on the local machine
            train_model(config)
        else:
            # submit jobs to the cluster
            python_cmd = f'python -c "import train; train.train_slurm(\'{config_file_path}\')"'
            job_n = config['experiment_name'] + '_' + config['model_name']
            output_path = os.path.join(ROOT_DIR, 'slurm_output')
            slurm_job_file = get_jobfile(python_cmd,
                                         job_n,
                                         sbatch_path=config['save_path'],
                                         output_path=output_path,
                                         partition=args.partition,
                                         cuda_module=CUDA_MODULE,
                                         conda_env=CONDA_ENV,
                                         )
            cp_process = subprocess.run(['sbatch', slurm_job_file],
                                        capture_output=True, check=True)
            cp_stdout = cp_process.stdout.decode()
            print(cp_stdout)

