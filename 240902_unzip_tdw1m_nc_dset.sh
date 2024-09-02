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
cd "$1"/images
mkdir "`basename $2 .zip`"
unzip -q $2 -d "`basename $2 .zip`"
echo "Unzipped $2 finished!"
