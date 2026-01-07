#!/bin/bash

# ARC-Challenge Activation Collection Script
# Usage: bash scripts/run_arc_collect.sh

# Set paths (modify as needed)
MODEL_PATH="/data1/shenth/models/llama/3.2-1b"
DATA_DIR="/data1/shenth/datasets/commonsense/origin/arc_c/"
OUTPUT_DIR="./outputs"

# Run collection
python -m src.main \
    --model_path $MODEL_PATH \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --max_samples -1 \
    --batch_size 4 \
    --max_length 256 \
    --dtype fp16 \
    --device_map cuda \
    --token_agg last \
    --template_name arc_mcq_v1 \
    --few_shot 0 \
    --seed 42 \
    --experiment_name arc_head_activation

# Alternative: use config file only
# python -m src.main --config configs/default.yaml

