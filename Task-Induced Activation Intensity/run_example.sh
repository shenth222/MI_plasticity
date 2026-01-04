#!/bin/bash

# 示例运行脚本
# 请根据您的实际路径修改以下变量

# 模型路径（修改为您的 Llama-3.2-1B 本地路径）
MODEL_PATH="/path/to/Llama-3.2-1B"

# 数据路径（修改为您的 CS170k 数据路径，或使用示例数据）
DATA_PATH="example_data.jsonl"

# 输出目录
OUTPUT_DIR="outputs/run_$(date +%Y%m%d_%H%M%S)"

# 运行参数
MAX_SAMPLES=1024
BATCH_SIZE=4
MAX_LENGTH=512
DEVICE="cuda:0"
DTYPE="fp16"
SEED=42

# 运行
python src/main.py \
  --model_path "$MODEL_PATH" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --max_samples $MAX_SAMPLES \
  --batch_size $BATCH_SIZE \
  --max_length $MAX_LENGTH \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --seed $SEED \
  --score_query_mode "last_token" \
  --norm_mode "zscore" \
  --lambda_ent 0.5 \
  --lambda_task 1.0

echo "运行完成！结果保存在: $OUTPUT_DIR"

