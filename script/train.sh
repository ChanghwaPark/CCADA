#!/bin/bash

dataset_root=${1}
gpu=${2}
src=${3}
tgt=${4}
alpha=0.9
temperature=0.05
network=resnet50
max_key_size=2048
min_conf_samples=3
max_iterations=5000
contrast_dim=256
for thresh in 0 0.9 0.95; do
  for cw in 0.1 0.5 1.0 2.0; do
    python main.py --src ${src} --tgt ${tgt} --gpu ${gpu} --cw ${cw} --thresh ${thresh} \
    --dataset_root ${dataset_root} --contrast_dim ${contrast_dim} --temperature ${temperature} \
    --alpha ${alpha} --network ${network} --max_key_size ${max_key_size} --min_conf_samples ${min_conf_samples} \
    --max_iterations ${max_iterations}
  done
done
