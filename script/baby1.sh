#!/bin/bash

python main.py --src visda_src --tgt visda_tgt --threshold 0.9 --tgt_weight 0.5 --contrast_weight 1.0 --confident_classes 10 --gpu 1 --lr_decay 1 --config_file config/config2.yml
python main.py --src visda_src --tgt visda_tgt --threshold 0.9 --tgt_weight 0.5 --contrast_weight 2.0 --confident_classes 10 --gpu 1 --lr_decay 1 --config_file config/config2.yml
python main.py --src visda_src --tgt visda_tgt --threshold 0.9 --tgt_weight 1.0 --contrast_weight 0.1 --confident_classes 10 --gpu 1 --lr_decay 1 --config_file config/config2.yml
python main.py --src visda_src --tgt visda_tgt --threshold 0.9 --tgt_weight 1.0 --contrast_weight 0.5 --confident_classes 10 --gpu 1 --lr_decay 1 --config_file config/config2.yml
python main.py --src visda_src --tgt visda_tgt --threshold 0.9 --tgt_weight 1.0 --contrast_weight 1.0 --confident_classes 10 --gpu 1 --lr_decay 1 --config_file config/config2.yml
python main.py --src visda_src --tgt visda_tgt --threshold 0.9 --tgt_weight 1.0 --contrast_weight 2.0 --confident_classes 10 --gpu 1 --lr_decay 1 --config_file config/config2.yml
python main.py --src visda_src --tgt visda_tgt --threshold 0.9 --tgt_weight 2.0 --contrast_weight 0.1 --confident_classes 10 --gpu 1 --lr_decay 1 --config_file config/config2.yml
python main.py --src visda_src --tgt visda_tgt --threshold 0.9 --tgt_weight 2.0 --contrast_weight 0.5 --confident_classes 10 --gpu 1 --lr_decay 1 --config_file config/config2.yml
python main.py --src visda_src --tgt visda_tgt --threshold 0.9 --tgt_weight 2.0 --contrast_weight 1.0 --confident_classes 10 --gpu 1 --lr_decay 1 --config_file config/config2.yml
python main.py --src visda_src --tgt visda_tgt --threshold 0.9 --tgt_weight 2.0 --contrast_weight 2.0 --confident_classes 10 --gpu 1 --lr_decay 1 --config_file config/config2.yml
