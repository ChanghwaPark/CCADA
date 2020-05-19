#!/bin/bash

gpu=${1}
thresh=${2}
max_key_size=${3}
contrast_dim=${4}
dataset_root=${5}
for temperature in 0.04 0.05 0.07; do
  for cw in 0.5 1.0 2.0; do
    for min_conf_samples in 1 3; do
      python main.py --src amazon --tgt dslr --gpu ${gpu} --cw ${cw} --thresh ${thresh} \
      --dataset_root ${dataset_root} --contrast_dim ${contrast_dim} --temperature ${temperature} \
      --alpha 0.9 --network resnet50 --max_key_size ${max_key_size} --min_conf_samples ${min_conf_samples} \
      --max_iterations 5000
    done
  done
done
