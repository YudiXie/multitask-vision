#!/bin/bash
for i in /om/user/yu_xie/data/tdw_images/*c_20240902
do
  for j in "$i"/images/*.zip
  do
    sbatch 240902_unzip_tdw1m_nc_dset.sh $i $j
  done
done
