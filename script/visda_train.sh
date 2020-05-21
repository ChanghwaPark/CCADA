#!/bin/bash

dataset_root=${1}
gpu=${2}
cw=${3}
thresh=${4}
for _ in 1 2 3; do
  for alpha in 0.9 0.99; do
      python main.py --src visda_src --tgt visda_tgt --gpu ${gpu} --cw ${cw} --thresh ${thresh} \
      --dataset_root ${dataset_root} --contrast_dim 256 --temperature 0.05 \
      --alpha ${alpha} --network resnet101 --max_key_size 16384 --min_conf_samples 3 \
      --max_iterations 50000
  done
done
