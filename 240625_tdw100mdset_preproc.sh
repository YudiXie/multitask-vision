#!/bin/bash
#SBATCH -t 30:00:00
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=normal
#SBATCH -e /om/weka/dicarlo/yu_xie/projects/multitask-vision/slurm_output/slurm-%j-240624_tdw100mdset.out
#SBATCH -o /om/weka/dicarlo/yu_xie/projects/multitask-vision/slurm_output/slurm-%j-240624_tdw100mdset.out

source ~/.bashrc
echo -e "System Info: \n----------\n$(hostnamectl)\n----------"
cd /om/user/yu_xie/projects/multitask-vision
conda activate mtvision
python tdw_dataset_preprocess.py --index /om/user/yu_xie/data/tdw_images/tdw_image_dataset_100m_20240222/index_img_100100440.csv
