#!/bin/bash

root_dir_name="/om/user/yu_xie/data/tdw_images/"

for file in \
  "tdw1m_2c_20240902/index_img_1350000.csv" \
  "tdw1m_4c_20240902/index_img_1350020.csv" \
  "tdw1m_6c_20240902/index_img_1349980.csv" \
  "tdw1m_8c_20240902/index_img_1350020.csv" \
  "tdw1m_16c_20240902/index_img_1350020.csv"
do
  csv_name="${root_dir_name}${file}"
  sbatch 240902_tdw1mncdset_preproc.sh $csv_name
done
