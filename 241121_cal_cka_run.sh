#!/bin/bash

for index in {0..60}
do
  sbatch 241121_cal_cka.sh $index
done
