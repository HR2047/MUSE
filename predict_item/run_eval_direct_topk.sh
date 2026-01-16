#!/bin/bash

DATASET_STATS_PATH="/home/hirosawa/research_m/MUSE/data/book_dataset_stats_direct.json"
CONFIG_PATH="/home/hirosawa/research_m/MUSE/predict_interest/configs/config_bookInterest_ver5.py"
LOG_DIR="log"

mkdir -p ${LOG_DIR}

for k in 1 2 3 4
do
    echo "Running evaluation with k=${k}"
    setsid nohup python -u evaluate_muse_direct_topk.py \
        --dataset_stats_path ${DATASET_STATS_PATH} \
        --config ${CONFIG_PATH} \
        --k ${k} \
        > ${LOG_DIR}/evaluate_muse_direct_topk_k${k}.log 2>&1 &
done

echo "All jobs submitted."
