#!/bin/bash

gpu=${1}
max_iterations=${2}
output_file=${3}

python validation.py --src art --tgt clipart --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/art_clipart_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.1_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_0/checkpoint_161_8695.weights
python validation.py --src art --tgt clipart --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/art_clipart_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.5_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_0/checkpoint_27_1436.weights
python validation.py --src art --tgt clipart --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/art_clipart_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_1.0_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_0/checkpoint_18_902.weights
python validation.py --src art --tgt clipart --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/art_clipart_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_2.0_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_0/checkpoint_13_725.weights
python validation.py --src art --tgt clipart --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/art_clipart_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.1_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_0/checkpoint_176_8800.weights
python validation.py --src art --tgt clipart --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/art_clipart_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.5_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_0/checkpoint_31_1550.weights
python validation.py --src art --tgt clipart --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/art_clipart_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_1.0_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_0/checkpoint_25_1250.weights
python validation.py --src art --tgt clipart --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/art_clipart_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_2.0_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_0/checkpoint_22_1134.weights
python validation.py --src art --tgt clipart --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/art_clipart_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.1_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_0/checkpoint_194_9700.weights
python validation.py --src art --tgt clipart --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/art_clipart_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.5_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_0/checkpoint_116_5800.weights
python validation.py --src art --tgt clipart --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/art_clipart_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_1.0_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_0/checkpoint_92_4600.weights
python validation.py --src art --tgt clipart --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/art_clipart_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_2.0_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_0/checkpoint_53_2650.weights
python validation.py --src art --tgt product --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/art_product_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.1_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_0/checkpoint_184_9200.weights
python validation.py --src art --tgt product --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/art_product_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.5_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_0/checkpoint_34_1700.weights
python validation.py --src art --tgt product --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/art_product_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_1.0_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_0/checkpoint_19_950.weights
python validation.py --src art --tgt product --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/art_product_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_2.0_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_0/checkpoint_20_1000.weights
python validation.py --src art --tgt product --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/art_product_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.1_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_0/checkpoint_198_9900.weights
python validation.py --src art --tgt product --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/art_product_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.5_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_0/checkpoint_103_5150.weights
python validation.py --src art --tgt product --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/art_product_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_1.0_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_0/checkpoint_18_3600.weights
python validation.py --src art --tgt product --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/art_product_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_2.0_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_0/checkpoint_54_2700.weights
python validation.py --src art --tgt product --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/art_product_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.1_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_0/checkpoint_191_9550.weights
python validation.py --src art --tgt product --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/art_product_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.5_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_0/checkpoint_88_4400.weights
python validation.py --src art --tgt product --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/art_product_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_1.0_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_0/checkpoint_156_7800.weights
python validation.py --src art --tgt product --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/art_product_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_2.0_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_0/checkpoint_35_1750.weights
python validation.py --src art --tgt real_world --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/art_real_world_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.1_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_0/checkpoint_142_9024.weights
python validation.py --src art --tgt real_world --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/art_real_world_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.5_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_0/checkpoint_28_1870.weights
python validation.py --src art --tgt real_world --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/art_real_world_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_1.0_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_0/checkpoint_13_876.weights
python validation.py --src art --tgt real_world --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/art_real_world_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_2.0_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_0/checkpoint_30_2311.weights
python validation.py --src art --tgt real_world --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/art_real_world_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.1_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_0/checkpoint_182_9606.weights
python validation.py --src art --tgt real_world --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/art_real_world_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.5_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_0/checkpoint_75_4286.weights
python validation.py --src art --tgt real_world --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/art_real_world_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_1.0_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_0/checkpoint_26_1458.weights
python validation.py --src art --tgt real_world --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/art_real_world_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_2.0_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_0/checkpoint_20_1098.weights
python validation.py --src art --tgt real_world --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/art_real_world_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.1_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_0/checkpoint_198_9800.weights
python validation.py --src art --tgt real_world --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/art_real_world_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.5_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_0/checkpoint_182_9100.weights
python validation.py --src art --tgt real_world --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/art_real_world_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_1.0_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_0/checkpoint_95_4750.weights
python validation.py --src art --tgt real_world --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/art_real_world_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_2.0_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_0/checkpoint_33_1650.weights
python validation.py --src clipart --tgt art --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/clipart_art_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.1_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_1/checkpoint_187_9350.weights
python validation.py --src clipart --tgt art --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/clipart_art_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.5_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_1/checkpoint_26_1300.weights
python validation.py --src clipart --tgt art --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/clipart_art_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_1.0_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_1/checkpoint_25_1250.weights
python validation.py --src clipart --tgt art --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/clipart_art_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_2.0_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_1/checkpoint_18_900.weights
python validation.py --src clipart --tgt art --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/clipart_art_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.1_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_1/checkpoint_192_9600.weights
python validation.py --src clipart --tgt art --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/clipart_art_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.5_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_1/checkpoint_63_3150.weights
python validation.py --src clipart --tgt art --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/clipart_art_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_1.0_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_1/checkpoint_24_1200.weights
python validation.py --src clipart --tgt art --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/clipart_art_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_2.0_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_1/checkpoint_22_1100.weights
python validation.py --src clipart --tgt art --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/clipart_art_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.1_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_1/checkpoint_193_9650.weights
python validation.py --src clipart --tgt art --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/clipart_art_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.5_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_1/checkpoint_142_7100.weights
python validation.py --src clipart --tgt art --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/clipart_art_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_1.0_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_1/checkpoint_118_5900.weights
python validation.py --src clipart --tgt art --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/clipart_art_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_2.0_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_1/checkpoint_64_3200.weights
python validation.py --src clipart --tgt product --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/clipart_product_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.1_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_1/checkpoint_195_9750.weights
python validation.py --src clipart --tgt product --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/clipart_product_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.5_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_1/checkpoint_92_4600.weights
python validation.py --src clipart --tgt product --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/clipart_product_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_1.0_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_1/checkpoint_33_1656.weights
python validation.py --src clipart --tgt product --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/clipart_product_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_2.0_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_1/checkpoint_46_2300.weights
python validation.py --src clipart --tgt product --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/clipart_product_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.1_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_1/checkpoint_199_9950.weights
python validation.py --src clipart --tgt product --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/clipart_product_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.5_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_1/checkpoint_99_4950.weights
python validation.py --src clipart --tgt product --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/clipart_product_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_1.0_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_1/checkpoint_39_1950.weights
python validation.py --src clipart --tgt product --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/clipart_product_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_2.0_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_1/checkpoint_39_1950.weights
python validation.py --src clipart --tgt product --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/clipart_product_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.1_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_1/checkpoint_191_9550.weights
python validation.py --src clipart --tgt product --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/clipart_product_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.5_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_1/checkpoint_97_4850.weights
python validation.py --src clipart --tgt product --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/clipart_product_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_1.0_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_1/checkpoint_64_3200.weights
python validation.py --src clipart --tgt product --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/clipart_product_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_2.0_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_1/checkpoint_52_2600.weights
python validation.py --src clipart --tgt real_world --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/clipart_real_world_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.1_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_1/checkpoint_144_9561.weights
python validation.py --src clipart --tgt real_world --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/clipart_real_world_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.5_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_1/checkpoint_53_3816.weights
python validation.py --src clipart --tgt real_world --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/clipart_real_world_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_1.0_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_1/checkpoint_30_1826.weights
python validation.py --src clipart --tgt real_world --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/clipart_real_world_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_2.0_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_1/checkpoint_21_1484.weights
python validation.py --src clipart --tgt real_world --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/clipart_real_world_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.1_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_1/checkpoint_193_9650.weights
python validation.py --src clipart --tgt real_world --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/clipart_real_world_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.5_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_1/checkpoint_62_3574.weights
python validation.py --src clipart --tgt real_world --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/clipart_real_world_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_1.0_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_1/checkpoint_32_1846.weights
python validation.py --src clipart --tgt real_world --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/clipart_real_world_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_2.0_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_1/checkpoint_144_9362.weights
python validation.py --src clipart --tgt real_world --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/clipart_real_world_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.1_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_1/checkpoint_196_9800.weights
python validation.py --src clipart --tgt real_world --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/clipart_real_world_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.5_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_1/checkpoint_144_7200.weights
python validation.py --src clipart --tgt real_world --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/clipart_real_world_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_1.0_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_1/checkpoint_141_7050.weights
python validation.py --src clipart --tgt real_world --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/clipart_real_world_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_2.0_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_1/checkpoint_116_5800.weights
python validation.py --src product --tgt art --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/product_art_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.1_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_2/checkpoint_132_6600.weights
python validation.py --src product --tgt art --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/product_art_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.5_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_2/checkpoint_20_1000.weights
python validation.py --src product --tgt art --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/product_art_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_1.0_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_2/checkpoint_21_1050.weights
python validation.py --src product --tgt art --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/product_art_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_2.0_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_2/checkpoint_10_500.weights
python validation.py --src product --tgt art --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/product_art_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.1_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_2/checkpoint_185_9200.weights
python validation.py --src product --tgt art --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/product_art_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.5_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_2/checkpoint_26_1300.weights
python validation.py --src product --tgt art --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/product_art_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_1.0_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_2/checkpoint_25_1250.weights
python validation.py --src product --tgt art --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/product_art_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_2.0_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_2/checkpoint_18_900.weights
python validation.py --src product --tgt art --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/product_art_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.1_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_2/checkpoint_165_8250.weights
python validation.py --src product --tgt art --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/product_art_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.5_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_2/checkpoint_37_1850.weights
python validation.py --src product --tgt art --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/product_art_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_1.0_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_2/checkpoint_42_2100.weights
python validation.py --src product --tgt art --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/product_art_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_2.0_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_2/checkpoint_97_4850.weights
python validation.py --src product --tgt clipart --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/product_clipart_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.1_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_2/checkpoint_118_6517.weights
python validation.py --src product --tgt clipart --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/product_clipart_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.5_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_2/checkpoint_28_1686.weights
python validation.py --src product --tgt clipart --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/product_clipart_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_1.0_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_2/checkpoint_15_961.weights
python validation.py --src product --tgt clipart --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/product_clipart_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_2.0_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_2/checkpoint_30_1512.weights
python validation.py --src product --tgt clipart --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/product_clipart_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.1_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_2/checkpoint_187_9406.weights
python validation.py --src product --tgt clipart --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/product_clipart_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.5_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_2/checkpoint_32_1600.weights
python validation.py --src product --tgt clipart --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/product_clipart_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_1.0_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_2/checkpoint_43_2350.weights
python validation.py --src product --tgt clipart --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/product_clipart_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_2.0_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_2/checkpoint_12_600.weights
python validation.py --src product --tgt clipart --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/product_clipart_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.1_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_2/checkpoint_191_9550.weights
python validation.py --src product --tgt clipart --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/product_clipart_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.5_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_2/checkpoint_74_3700.weights
python validation.py --src product --tgt clipart --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/product_clipart_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_1.0_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_2/checkpoint_108_5400.weights
python validation.py --src product --tgt clipart --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/product_clipart_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_2.0_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_2/checkpoint_91_4550.weights
python validation.py --src product --tgt real_world --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/product_real_world_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.1_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_2/checkpoint_127_9392.weights
python validation.py --src product --tgt real_world --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/product_real_world_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.5_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_2/checkpoint_43_2762.weights
python validation.py --src product --tgt real_world --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/product_real_world_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_1.0_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_2/checkpoint_21_1457.weights
python validation.py --src product --tgt real_world --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/product_real_world_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_2.0_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_2/checkpoint_16_1204.weights
python validation.py --src product --tgt real_world --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/product_real_world_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.1_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_2/checkpoint_147_9302.weights
python validation.py --src product --tgt real_world --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/product_real_world_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.5_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_2/checkpoint_118_8028.weights
python validation.py --src product --tgt real_world --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/product_real_world_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_1.0_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_2/checkpoint_110_6751.weights
python validation.py --src product --tgt real_world --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/product_real_world_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_2.0_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_2/checkpoint_127_8994.weights
python validation.py --src product --tgt real_world --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/product_real_world_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.1_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_2/checkpoint_199_9950.weights
python validation.py --src product --tgt real_world --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/product_real_world_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.5_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_2/checkpoint_131_6550.weights
python validation.py --src product --tgt real_world --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/product_real_world_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_1.0_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_2/checkpoint_77_3850.weights
python validation.py --src product --tgt real_world --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/product_real_world_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_2.0_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_2/checkpoint_72_3600.weights
python validation.py --src real_world --tgt art --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/real_world_art_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.1_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_3/checkpoint_106_5300.weights
python validation.py --src real_world --tgt art --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/real_world_art_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.5_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_3/checkpoint_20_1000.weights
python validation.py --src real_world --tgt art --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/real_world_art_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_1.0_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_3/checkpoint_18_900.weights
python validation.py --src real_world --tgt art --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/real_world_art_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_2.0_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_3/checkpoint_12_600.weights
python validation.py --src real_world --tgt art --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/real_world_art_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.1_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_3/checkpoint_133_6650.weights
python validation.py --src real_world --tgt art --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/real_world_art_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.5_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_3/checkpoint_30_1500.weights
python validation.py --src real_world --tgt art --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/real_world_art_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_1.0_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_3/checkpoint_26_1300.weights
python validation.py --src real_world --tgt art --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/real_world_art_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_2.0_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_3/checkpoint_19_950.weights
python validation.py --src real_world --tgt art --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/real_world_art_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.1_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_3/checkpoint_195_9750.weights
python validation.py --src real_world --tgt art --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/real_world_art_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.5_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_3/checkpoint_86_4300.weights
python validation.py --src real_world --tgt art --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/real_world_art_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_1.0_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_3/checkpoint_42_2100.weights
python validation.py --src real_world --tgt art --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/real_world_art_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_2.0_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_3/checkpoint_38_1900.weights
python validation.py --src real_world --tgt clipart --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/real_world_clipart_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.1_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_3/checkpoint_53_2650.weights
python validation.py --src real_world --tgt clipart --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/real_world_clipart_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.5_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_3/checkpoint_28_1400.weights
python validation.py --src real_world --tgt clipart --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/real_world_clipart_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_1.0_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_3/checkpoint_39_1950.weights
python validation.py --src real_world --tgt clipart --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/real_world_clipart_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_2.0_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_3/checkpoint_17_850.weights
python validation.py --src real_world --tgt clipart --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/real_world_clipart_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.1_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_3/checkpoint_140_7000.weights
python validation.py --src real_world --tgt clipart --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/real_world_clipart_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.5_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_3/checkpoint_54_2700.weights
python validation.py --src real_world --tgt clipart --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/real_world_clipart_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_1.0_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_3/checkpoint_27_1350.weights
python validation.py --src real_world --tgt clipart --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/real_world_clipart_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_2.0_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_3/checkpoint_23_1150.weights
python validation.py --src real_world --tgt clipart --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/real_world_clipart_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.1_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_3/checkpoint_189_9450.weights
python validation.py --src real_world --tgt clipart --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/real_world_clipart_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.5_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_3/checkpoint_97_4850.weights
python validation.py --src real_world --tgt clipart --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/real_world_clipart_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_1.0_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_3/checkpoint_65_3250.weights
python validation.py --src real_world --tgt clipart --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/real_world_clipart_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_2.0_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_3/checkpoint_61_3050.weights
python validation.py --src real_world --tgt product --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/real_world_product_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.1_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_3/checkpoint_199_9950.weights
python validation.py --src real_world --tgt product --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/real_world_product_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.5_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_3/checkpoint_29_1450.weights
python validation.py --src real_world --tgt product --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/real_world_product_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_1.0_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_3/checkpoint_21_1050.weights
python validation.py --src real_world --tgt product --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/real_world_product_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_2.0_thresh_0.0_max_key_size_2048_min_conf_samples_3_gpu_3/checkpoint_42_2100.weights
python validation.py --src real_world --tgt product --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/real_world_product_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.1_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_3/checkpoint_190_9738.weights
python validation.py --src real_world --tgt product --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/real_world_product_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.5_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_3/checkpoint_99_5076.weights
python validation.py --src real_world --tgt product --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/real_world_product_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_1.0_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_3/checkpoint_35_1800.weights
python validation.py --src real_world --tgt product --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/real_world_product_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_2.0_thresh_0.9_max_key_size_2048_min_conf_samples_3_gpu_3/checkpoint_52_2990.weights
python validation.py --src real_world --tgt product --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/real_world_product_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.1_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_3/checkpoint_189_9450.weights
python validation.py --src real_world --tgt product --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/real_world_product_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_0.5_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_3/checkpoint_190_9500.weights
python validation.py --src real_world --tgt product --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/real_world_product_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_1.0_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_3/checkpoint_126_6300.weights
python validation.py --src real_world --tgt product --gpu ${gpu} --network resnet50 --max_iterations ${max_iterations} --output_file ${output_file} \
--model_dir logs/real_world_product_resnet50_contrast_dim_256_temperature_0.05_alpha_0.9_cw_2.0_thresh_0.95_max_key_size_2048_min_conf_samples_3_gpu_3/checkpoint_70_3500.weights
