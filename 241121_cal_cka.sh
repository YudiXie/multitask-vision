#!/bin/bash
#SBATCH -t 6:00:00
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=normal
#SBATCH -e /om/weka/dicarlo/yu_xie/projects/multitask-vision/slurm_output/slurm-%j-cal_cka.out
#SBATCH -o /om/weka/dicarlo/yu_xie/projects/multitask-vision/slurm_output/slurm-%j-cal_cka.out

source ~/.bashrc
echo -e "System Info: \n----------\n$(hostnamectl)\n----------"
cd /om/user/yu_xie/projects/multitask-vision
conda activate mtvision3
python calculate_cka.py --do $1
echo "calculate CKA index $1 finished!"
