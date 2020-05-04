#!/bin/bash

gpu=${1}
cw=${2}
thresh=${3}
dataset_root=${4}
for contrast_normalize in false true; do
  for pseudo_labeling in 'info' 'kmeans' 'classifier'; do
    if [[ "$pseudo_labeling" == "info" ]]; then
      for pseudo_normalize in false true; do
        python main.py --src dslr --tgt amazon --gpu ${gpu} --cw ${cw} --thresh ${thresh} \
        --dataset_root ${dataset_root} --min_conf_classes 18 --contrast_normalize ${contrast_normalize} \
        --pseudo_labeling ${pseudo_labeling} --pseudo_normalize ${pseudo_normalize}
      done
    else
      python main.py --src dslr --tgt amazon --gpu ${gpu} --cw ${cw} --thresh ${thresh} \
      --dataset_root ${dataset_root} --min_conf_classes 18 --contrast_normalize ${contrast_normalize} \
      --pseudo_labeling ${pseudo_labeling}
    fi
  done
done
