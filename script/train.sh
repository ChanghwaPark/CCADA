#!/bin/bash

gpu=${1}
cw=${2}
contrast_dim=${3}
dataset_root=${4}
for thresh in 0.0 0.9 0.95; do
  python main.py --src visda_src --tgt visda_tgt --gpu ${gpu} --cw ${cw} --thresh ${thresh} \
  --dataset_root ${dataset_root} --min_conf_classes 8 --contrast_dim ${contrast_dim} \
  --alpha 0.9 --network resnet101 --gamma 0.0005 --decay_rate 2.25
done
