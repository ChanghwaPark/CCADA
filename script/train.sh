#!/bin/bash

dataset_root='/home/omega/datasets'
gpu=${1}
src_1=${2}
tgt_1=${3}
src_2=${4}
tgt_2=${5}
src_3=${6}
tgt_3=${7}
alpha=0.9
temperature=0.05
network=resnet50
max_key_size=2048
min_conf_samples=3
max_iterations=10000
contrast_dim=256
#thresh=0.95
for thresh in 0.0 0.9 0.95; do
  for cw in 0.1 0.5 1.0 2.0; do
    for _ in 1 2; do
      python main.py --src ${src_1} --tgt ${tgt_1} --gpu ${gpu} --cw ${cw} --thresh ${thresh} \
      --dataset_root ${dataset_root} --contrast_dim ${contrast_dim} --temperature ${temperature} \
      --alpha ${alpha} --network ${network} --max_key_size ${max_key_size} --min_conf_samples ${min_conf_samples} \
      --max_iterations ${max_iterations}
    done
  done
done

for thresh in 0.0 0.9 0.95; do
  for cw in 0.1 0.5 1.0 2.0; do
    for _ in 1 2; do
      python main.py --src ${src_2} --tgt ${tgt_2} --gpu ${gpu} --cw ${cw} --thresh ${thresh} \
      --dataset_root ${dataset_root} --contrast_dim ${contrast_dim} --temperature ${temperature} \
      --alpha ${alpha} --network ${network} --max_key_size ${max_key_size} --min_conf_samples ${min_conf_samples} \
      --max_iterations ${max_iterations}
    done
  done
done

for thresh in 0.0 0.9 0.95; do
  for cw in 0.1 0.5 1.0 2.0; do
    for _ in 1 2; do
      python main.py --src ${src_3} --tgt ${tgt_3} --gpu ${gpu} --cw ${cw} --thresh ${thresh} \
      --dataset_root ${dataset_root} --contrast_dim ${contrast_dim} --temperature ${temperature} \
      --alpha ${alpha} --network ${network} --max_key_size ${max_key_size} --min_conf_samples ${min_conf_samples} \
      --max_iterations ${max_iterations}
    done
  done
done
