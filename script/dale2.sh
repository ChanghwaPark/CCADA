#!/bin/bash

python main.py --src visda_src --tgt visda_tgt --tgt_weight 0.0 --contrast_weight 0.1 --confident_classes 10 --gpu 2 --max_key_feature_size 16384
python main.py --src visda_src --tgt visda_tgt --tgt_weight 0.0 --contrast_weight 0.5 --confident_classes 10 --gpu 2 --max_key_feature_size 16384
