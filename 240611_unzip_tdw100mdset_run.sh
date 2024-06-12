#!/bin/bash
for i in /om/user/yu_xie/data/tdw_images/tdw_image_dataset_100m_20240222/images/*.zip
do
  sbatch 240611_unzip_tdw100mdset.sh $i
done
