#!/bin/bash
#SBATCH -t 23:00:00
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --mem=32G
#SBATCH --partition=dicarlo
#SBATCH -e /om/weka/dicarlo/yu_xie/projects/multitask-vision/slurm_output/slurm-%j-240613_unzip_tdw100mdset.out
#SBATCH -o /om/weka/dicarlo/yu_xie/projects/multitask-vision/slurm_output/slurm-%j-240613_unzip_tdw100mdset.out

source ~/.bashrc
echo -e "System Info: \n----------\n$(hostnamectl)\n----------"
cd /om/user/yu_xie/data/tdw_images/tdw_image_dataset_100m_20240222/images
rm -rf lava_field
mkdir "lava_field"
unzip -q "/om/user/yu_xie/data/tdw_images/tdw_image_dataset_100m_20240222/images/lava_field.zip" -d "/om/user/yu_xie/data/tdw_images/tdw_image_dataset_100m_20240222/images/lava_field"
echo "Unzipped lava_field finished!"
