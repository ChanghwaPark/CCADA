#!/bin/bash

PROJ_ROOT="/home/omega/mycodes/dass_pytorch"
LOG_ROOT_DIR="logs"
PROJ_NAME="amazon_dslr_ResNet50_gpu_2"
LOG_FOLDER="${PROJ_ROOT}/${LOG_ROOT_DIR}/${PROJ_NAME}"
LOG_FILE="${LOG_FOLDER}/stdout.log"

python ${PROJ_ROOT}/main.py \
--dataset office \
--src amazon \
--tgt dslr \
--gpu 2 \
| tee "${LOG_FILE}"
