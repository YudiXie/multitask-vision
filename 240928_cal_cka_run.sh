#!/bin/bash

for index in {0..83}
do
  sbatch 240928_cal_cka.sh $index
done
