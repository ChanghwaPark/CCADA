#!/bin/bash

gpu=${1}
cw=${2}
contrast_dim=${3}
dataset_root=${4}
for alpha in 0.9 0.99 0.999; do
  for thresh in 0.0 0.5 0.95; do
    python main.py --src dslr --tgt amazon --gpu ${gpu} --cw ${cw} --thresh ${thresh} \
    --dataset_root ${dataset_root} --min_conf_classes 18 --contrast_dim ${contrast_dim} \
    --alpha ${alpha}
  done
done
