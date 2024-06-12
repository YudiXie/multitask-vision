#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=normal
#SBATCH -e /om/weka/dicarlo/yu_xie/projects/multitask-vision/slurm_output/slurm-%j-240611_unzip_tdw100mdset.out
#SBATCH -o /om/weka/dicarlo/yu_xie/projects/multitask-vision/slurm_output/slurm-%j-240611_unzip_tdw100mdset.out

source ~/.bashrc
echo -e "System Info: \n----------\n$(hostnamectl)\n----------"
cd /om/user/yu_xie/data/tdw_images/tdw_image_dataset_100m_20240222/images
mkdir "`basename $1 .zip`"
unzip -q $1 -d "`basename $1 .zip`"
echo "Unzipped $1 finished!"
