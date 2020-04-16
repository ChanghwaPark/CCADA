#!/bin/bash

python main.py --src visda_src --tgt visda_tgt --tgt_weight 0.0 --contrast_weight 1.0 --confident_classes 10 --gpu 1 --max_key_feature_size 16384 --tclip 1
python main.py --src visda_src --tgt visda_tgt --tgt_weight 0.0 --contrast_weight 2.0 --confident_classes 10 --gpu 1 --max_key_feature_size 16384 --tclip 1
python main.py --src visda_src --tgt visda_tgt --tgt_weight 0.0 --contrast_weight 1.0 --confident_classes 10 --gpu 1 --max_key_feature_size 16384 --tclip 5
python main.py --src visda_src --tgt visda_tgt --tgt_weight 0.0 --contrast_weight 2.0 --confident_classes 10 --gpu 1 --max_key_feature_size 16384 --tclip 5
python main.py --src visda_src --tgt visda_tgt --tgt_weight 0.0 --contrast_weight 1.0 --confident_classes 10 --gpu 1 --max_key_feature_size 16384 --tclip 10
python main.py --src visda_src --tgt visda_tgt --tgt_weight 0.0 --contrast_weight 2.0 --confident_classes 10 --gpu 1 --max_key_feature_size 16384 --tclip 10
