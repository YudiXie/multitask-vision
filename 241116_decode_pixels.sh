#!/bin/bash
#SBATCH -t 8:00:00
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=dicarlo
#SBATCH -e /om/weka/dicarlo/yu_xie/projects/multitask-vision/slurm_output/slurm-%j-decode_pixels.out
#SBATCH -o /om/weka/dicarlo/yu_xie/projects/multitask-vision/slurm_output/slurm-%j-decode_pixels.out

source ~/.bashrc
echo -e "System Info: \n----------\n$(hostnamectl)\n----------"
cd /om/user/yu_xie/projects/multitask-vision
conda activate mtvision3
python pixel_decode.py
echo "decode pixels finished!"
