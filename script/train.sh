#!/bin/bash

gpu=${1}
temperature=${2}
lr=${3}
dataset_root=${4}
for cw in 0.3 0.5 0.7; do
  for gamma in 0.0005 0.001 0.002; do
    for decay_rate in 2.25 1.5 0.75 3.0; do
      python main.py --src visda_src --tgt visda_tgt --gpu ${gpu} --cw ${cw} --thresh 0.9 \
      --dataset_root ${dataset_root} --contrast_dim 256 --temperature ${temperature} --lr ${lr}\
      --alpha 0.9 --network resnet101 --gamma ${gamma} --decay_rate ${decay_rate} --max_iterations 10000
    done
  done
done
